"""
subjectivity_scoring.py — Phase 3: score transcripts with the fine-tuned multi-task
SubjECTive-QA model and aggregate to a per-transcript feature dict.

This is to the Phase-2 checkpoint what sentiment_scoring.py is to FinBERT: a scorer class
(`SubjectivityScorer`) plus a transcript-level aggregator (`score_transcript_subjectivity`)
and an on-disk cache, so backtest.py can pull cached features instead of re-running the GPU.

Pipeline per transcript:
  1. split_transcript -> Q&A half (preprocessing.py),
  2. segment_qa_pairs -> [{question, answer}, ...] (Phase 1; segmentation quality is ITS
     responsibility — the smoke test below surfaces a 0-pair break immediately),
  3. score each pair on the six dimensions as a CONTINUOUS 0-2 expectation (no argmax),
  4. aggregate across the call's answers into a flat feature dict.

Two aggregations per dimension (12 features total):
  * {dim}_mean       — mean continuous score across answers (the level).
  * frac_low_{dim}   — fraction of answers scoring below LOW_CUTOFF toward 0. This is the
                       tail the mean washes out: frac_low_clear (unclear answers) and
                       frac_low_relevant (evasive non-answers) are the thesis, but we emit
                       the tail for all six so Phase 4 can decide which carry the edge.

The FinBERT cache (backtest_scores.json) is untouched; its composite stays a feature.
Cache lives in its own file (subjectivity_scores.json), keyed "SYM:year:Q" exactly like
backtest_scores.json.

Smoke test:
    python subjectivity_scoring.py AAPL                  # default model dir
    python subjectivity_scoring.py AAPL subjectivity_model
"""

import json
import os
import sys

import numpy as np
import torch

import preprocessing as pp          # reused: split_transcript, segment_qa_pairs

# --- config ---

N_CLASSES = 3                       # 0 = neg-demonstrative, 1 = neutral, 2 = pos-demonstrative
LOW_CUTOFF = 0.8                    # an answer counts as "low" on a dim if its 0-2 score < this
DEFAULT_MODEL_DIR = "subjectivity_model"
SUBJECTIVITY_CACHE_PATH = "subjectivity_scores.json"


class SubjectivityScorer:
    """Inference wrapper over the multi-task SubjECTive-QA checkpoint, mirroring
    FinBERTScorer's shape (device/amp handling, batched fp32-softmax inference)."""

    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR, batch_size: int = 64):
        # Lazy import: only constructing a scorer needs the model class (and its `datasets`
        # dependency, which is training-only). Keeps `import subjectivity_scoring` cheap.
        from transformers import AutoTokenizer
        from train_subjectivity import MultiTaskSubjectivityModel

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # load_trained returns the model already .to(device).eval(), plus tokenizer + meta.
        self.model, _loaded_tok, self.meta = MultiTaskSubjectivityModel.load_trained(
            model_dir, device=str(self.device)
        )
        # CRITICAL: tokenize from the SAME source training used. train_subjectivity.run()
        # uses AutoTokenizer.from_pretrained(base_model), but save() never persists a
        # tokenizer, so load_trained's AutoTokenizer.from_pretrained(out_dir) resolves a
        # different (degenerate) tokenizer that maps every pair to identical ids → constant
        # scores. Re-load from meta["base_model"] so inference matches training exactly.
        self.tokenizer = AutoTokenizer.from_pretrained(self.meta["base_model"])
        self.dimensions = list(self.meta["dimensions"])     # uppercase, canonical order
        # max_length MUST come from the checkpoint, not a hardcoded 512 — it has to match
        # the value the pairs were tokenized with during training, or inference drifts.
        self.max_length = int(self.meta.get("max_length", 256))
        self.batch_size = batch_size
        # FinBERT is inference-only here; run the forward pass in fp16 on GPU for throughput.
        # The expectation is computed from an fp32 softmax, so the autocast dtype is
        # immaterial to the result; CPU has no fp16 fast path, so gate AMP on CUDA only.
        self.use_amp = self.device.type == "cuda"
        self._validate_label_scheme()
        # precompute label weights [0, 1, 2] for the E[label] dot product
        self._weights = torch.arange(N_CLASSES, dtype=torch.float32, device=self.device)

    def _validate_label_scheme(self):
        """E[label] = Σ p_k·k assumes softmax column j == integer label j, with
        2 = 'positively demonstrative' (high) and 0 = 'negatively demonstrative' (low).
        If a future relabel reorders the classes the continuous score inverts SILENTLY —
        so fail loudly instead of producing plausible-but-backwards features."""
        scheme = self.meta.get("label_scheme")
        if not scheme:
            return  # older checkpoint without the field; trust the canonical 0/1/2 order
        expected = {"0": "negative", "1": "neutral", "2": "positive"}
        for k, kw in expected.items():
            if k not in scheme or kw not in str(scheme[k]).lower():
                raise ValueError(
                    f"Unexpected label_scheme {scheme}; scoring assumes 0=neg, 1=neutral, "
                    "2=pos. Adjust the E[label] math if the checkpoint's scheme changed."
                )

    @torch.inference_mode()
    def score_pairs(self, pairs: list[dict]) -> list[dict]:
        """Continuous 0-2 score per dimension for each (question, answer) pair.

        No argmax: per dimension we take the probability-weighted expectation
        E[label] = Σ_k p_k · k over the 3-class softmax (the k=0 term drops out), giving a
        smooth 0-2 score where 2 ≈ strongly positively demonstrative on that dimension.
        Returns one {DIM -> score} dict per input pair, in order.
        """
        if not pairs:
            return []
        results: list[dict] = [dict() for _ in pairs]
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i:i + self.batch_size]
            questions = [(p.get("question") or "") for p in batch]
            answers = [(p.get("answer") or "") for p in batch]
            # Tokenize EXACTLY as training did: question and answer as a TEXT PAIR (two
            # args), so the encoder inserts its pair separator (</s></s> for RoBERTa).
            # Concatenating into one string, or hardcoding max_length, distribution-shifts
            # inference away from training — the single biggest correctness trap here.
            enc = self.tokenizer(
                questions, answers, truncation=True, max_length=self.max_length,
                padding=True, return_tensors="pt",
            ).to(self.device)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16,
                                enabled=self.use_amp):
                logits = self.model(enc["input_ids"], enc["attention_mask"])
            for d in self.dimensions:
                # softmax in fp32 for numerically stable probabilities (logits may be fp16)
                probs = torch.softmax(logits[d].float(), dim=-1)        # (B, 3)
                exp = (probs * self._weights).sum(dim=-1).cpu().numpy()  # (B,) in [0, 2]
                for j, v in enumerate(exp):
                    results[i + j][d] = float(v)
        return results


# --- transcript-level aggregation ---

def _empty_features(dimensions: list[str]) -> dict:
    """Feature dict with every value None — for calls with no parseable Q&A. We return
    None rather than a fake-neutral 1.0 so the Phase-4 feature layer drops/imputes
    explicitly instead of training on laundered missing data."""
    feats = {}
    for d in dimensions:
        dl = d.lower()
        feats[f"{dl}_mean"] = None
        feats[f"frac_low_{dl}"] = None
    return feats


def score_transcript_subjectivity(qa_pairs: list[dict], ticker: str, date: str,
                                  scorer: SubjectivityScorer,
                                  low_cutoff: float = LOW_CUTOFF) -> dict:
    """Aggregate per-pair continuous scores into one transcript-level feature dict.

    Analogous to sentiment_scoring.score_transcript. Aggregation is over the call's
    answers (one pair == one management answer):
      {dim}_mean       — mean 0-2 score across answers (the level),
      frac_low_{dim}   — fraction of answers scoring < low_cutoff (the evasive/unclear tail).
    Returns {ticker, date, n_pairs, features}. n_pairs lets Phase 4 weight or drop thin calls.
    """
    dims = scorer.dimensions
    if not qa_pairs:
        return {"ticker": ticker, "date": date, "n_pairs": 0,
                "features": _empty_features(dims)}

    scored = scorer.score_pairs(qa_pairs)                    # [{DIM -> 0..2}, ...]
    arrs = {d: np.asarray([s[d] for s in scored], dtype=float) for d in dims}

    features = {}
    for d in dims:
        a = arrs[d]
        dl = d.lower()
        features[f"{dl}_mean"] = float(a.mean())
        features[f"frac_low_{dl}"] = float((a < low_cutoff).mean())

    return {"ticker": ticker, "date": date, "n_pairs": len(scored), "features": features}


# --- cache (own file; FinBERT's backtest_scores.json stays untouched) ---

def load_subjectivity_cache(path: str = SUBJECTIVITY_CACHE_PATH) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_subjectivity_cache(cache: dict, path: str = SUBJECTIVITY_CACHE_PATH) -> None:
    with open(path, "w") as f:
        json.dump(cache, f)


def subjectivity_features_for(client, scorer, cache, symbol, year, quarter,
                              low_cutoff: float = LOW_CUTOFF) -> dict | None:
    """Aggregated subjectivity features for one transcript, cached. Mirrors
    backtest.composite_for: key "SYM:year:Q", fail-soft to None on missing transcript.

    NB: we deliberately do NOT clean_fmp_text the Q&A before segmenting — segment_qa_pairs
    relies on line breaks (splitlines) and whitespace-collapsing would destroy them.
    """
    key = f"{symbol}:{year}:{quarter}"
    if key in cache:
        return cache[key]

    t = client.get_transcript(symbol, year, quarter)
    if not t or not t["content"]:
        cache[key] = None
        return None

    tdate = t["dt"].isoformat() if t.get("dt") else str(year)
    split = pp.split_transcript(t["content"])
    qa_pairs = pp.segment_qa_pairs(split["qa"])              # raw Q&A; newlines intact

    record = score_transcript_subjectivity(qa_pairs, symbol, tdate, scorer,
                                           low_cutoff=low_cutoff)
    cache[key] = record
    return record


# --- smoke test ---

def main():
    sym = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    model_dir = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL_DIR
    print(f"== subjectivity smoke test for {sym} (model dir: {model_dir}) ==")

    from fmp_client import FMPClient

    scorer = SubjectivityScorer(model_dir)
    print(f"checkpoint loaded: device={scorer.device} max_length={scorer.max_length}")
    print(f"  dimensions: {scorer.dimensions}")

    client = FMPClient()
    dates = client.transcript_dates(sym)
    if not dates:
        print("No transcripts found.")
        return
    last = dates[-1]
    t = client.get_transcript(sym, last["year"], last["quarter"])
    if not t or not t["content"]:
        print("Empty transcript.")
        return

    split = pp.split_transcript(t["content"])
    pairs = pp.segment_qa_pairs(split["qa"])
    print(f"{last['year']}Q{last['quarter']}: {len(pairs)} Q&A pairs segmented.")
    if not pairs:
        print("!! 0 pairs — Q&A segmentation (Phase 1) is broken for this format. "
              "Fix segment_qa_pairs before scoring anything.")
        return

    rec = score_transcript_subjectivity(pairs, sym, str(last["year"]), scorer)
    print(f"n_pairs={rec['n_pairs']}")
    print("aggregated features:")
    for k, v in rec["features"].items():
        print(f"  {k:<22} {v:.4f}")

    # eyeball the first few per-pair scores; every value must sit in [0, 2].
    sample = scorer.score_pairs(pairs[:3])
    print("first 3 pairs (continuous 0-2 per dimension):")
    for i, s in enumerate(sample):
        cells = "  ".join(f"{d.lower()[:4]}={s[d]:.2f}" for d in scorer.dimensions)
        print(f"  pair {i}: {cells}")
    flat = [s[d] for s in sample for d in scorer.dimensions]
    if flat and (min(flat) < 0.0 or max(flat) > 2.0):
        print(f"  !! out-of-range score detected (min={min(flat):.3f} max={max(flat):.3f}) "
              "— check the softmax/expectation math.")


if __name__ == "__main__":
    main()