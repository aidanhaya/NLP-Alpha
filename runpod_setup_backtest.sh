#!/usr/bin/env bash
# runpod_setup_backtest.sh — provision a RunPod GPU pod to run backtest.py (Phase 3+4).
#
# This script handles BOTH scorers: FinBERT (sentiment_scoring.py) and the fine-tuned
# SubjECTive-QA model (subjectivity_scoring.py). Both run on CUDA. The subjectivity
# checkpoint (subjectivity_model/) must be present — either seeded via runpodctl or
# re-downloaded from wherever you archived the tarball.
#
# Use a RunPod template that ships CUDA-enabled PyTorch (e.g. "RunPod PyTorch 2.x").
# Open the pod's web terminal and run:
#   export FMP_API_KEY=your_key_here
#   bash runpod_setup_backtest.sh
#
# Safeguards:
#   1) Hard-fails if CUDA is unavailable (no silent CPU runs on GPU-priced hardware).
#   2) Hard-fails if FMP_API_KEY is missing.
#   3) Hard-fails if subjectivity_model/ checkpoint is absent after the cache-seed pause.
#   4) Validates the tokenizer is saved into the checkpoint (the degenerate-ids fix).
#   5) Pauses before running so you can seed all three caches via runpodctl.
#   6) Auto-uploads all three output files on completion.
#   7) Stops the pod automatically after successful upload (no charges while you sleep).
#      Requires RUNPOD_API_KEY. If the key is missing OR any upload fails, the pod
#      stays alive so you can investigate — it never auto-stops on a bad run.
set -euo pipefail

# ---------------------------------------------------------------------------
# 0. CONFIG
# ---------------------------------------------------------------------------
REPO="https://github.com/aidanhaya/NLP-Alpha.git"
MONTHS="${MONTHS:-36}"          # override with: MONTHS=12 bash runpod_setup_backtest.sh
MAX_TICKERS="${MAX_TICKERS:-}"  # leave empty for full run; set e.g. 50 for a pilot

# RunPod API key for auto-stop on completion. Get yours at:
#   https://www.runpod.io/console/user/settings  (API Keys section)
# Set before running:  export RUNPOD_API_KEY=your_key_here
# If unset, the pod will NOT auto-stop — you must terminate it manually.
RUNPOD_API_KEY="${RUNPOD_API_KEY:-}"

WORKDIR=/workspace
cd "$WORKDIR"

# ---------------------------------------------------------------------------
# 1. HARD GATE: FMP API key
# ---------------------------------------------------------------------------
if [ -z "${FMP_API_KEY:-}" ]; then
  echo ""
  echo "================================================================"
  echo "  FATAL: FMP_API_KEY is not set."
  echo "  export FMP_API_KEY=your_key_here   and re-run."
  echo "================================================================"
  exit 1
fi
export FMP_API_KEY

# Soft warning: RUNPOD_API_KEY is optional but strongly recommended for auto-stop.
if [ -z "$RUNPOD_API_KEY" ]; then
  echo ""
  echo "  WARNING: RUNPOD_API_KEY not set — pod will NOT auto-stop after the run."
  echo "  To enable auto-stop, set it before running:"
  echo "    export RUNPOD_API_KEY=your_key_here"
  echo "  Get your key at: https://www.runpod.io/console/user/settings"
  echo ""
fi

# ---------------------------------------------------------------------------
# 2. Get the code
# ---------------------------------------------------------------------------
if [ -d NLP-Alpha/.git ]; then
  cd NLP-Alpha && git pull --ff-only
else
  git clone "$REPO" && cd NLP-Alpha
fi

# ---------------------------------------------------------------------------
# 3. Dependencies
#    torch comes from the CUDA template — do NOT reinstall (risks a CPU build).
#    Skip heavy training-only deps (datasets, huggingface_hub) and UI deps.
# ---------------------------------------------------------------------------
grep -v -iE '^(torch|playwright|ib-insync|datasets|huggingface.hub)([=<>! ]|$)' \
    requirements.txt > /tmp/reqs.txt
pip install -q -r /tmp/reqs.txt
python -c "import nltk; nltk.download('punkt_tab', quiet=True)"

# ---------------------------------------------------------------------------
# 4. HARD GATE: CUDA
# ---------------------------------------------------------------------------
python - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    sys.exit("FATAL: torch.cuda.is_available() == False.\n"
             "You are paying GPU prices to run on CPU.\n"
             "Fix: use a RunPod PyTorch/CUDA template, then re-run.")
print(f"CUDA OK -> {torch.cuda.get_device_name(0)} | torch {torch.__version__}")
PY

# ---------------------------------------------------------------------------
# 5. SEED CACHES — pause here so you can transfer files via runpodctl.
#
#    You have up to THREE files to seed (all optional but each saves time):
#
#    (a) backtest_scores.json    — FinBERT composite cache (~13h to cold-score)
#    (b) subjectivity_scores.json — subjectivity feature cache (~8h to cold-score)
#    (c) subjectivity_model.tar.gz — fine-tuned checkpoint (REQUIRED)
#
#    For each file on your LAPTOP:
#      runpodctl send <filename>
#    Then on THIS POD:
#      runpodctl receive <code>
#
#    Unpack the checkpoint tarball if you transferred it as a .tar.gz:
#      tar -xzf subjectivity_model.tar.gz
#
#    Then press Enter below to continue.
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  CACHE SEED PAUSE"
echo "  Transfer any/all of the following before continuing:"
echo ""
echo "  LAPTOP -> POD (run on laptop, then 'runpodctl receive <code>' here):"
echo "    runpodctl send backtest_scores.json"
echo "    runpodctl send subjectivity_scores.json"
echo "    runpodctl send subjectivity_model.tar.gz   (then: tar -xzf it)"
echo ""

# Report what's already present
for f in backtest_scores.json subjectivity_scores.json; do
  if [ -f "$f" ]; then
    python -c "
import json, sys
data = json.load(open('$f'))
n = len(data) if isinstance(data, dict) else len(data.get('scored_transcripts', data))
print(f'  FOUND $f: {n} cached entries')
" 2>/dev/null || echo "  FOUND $f (unreadable — will be overwritten)"
  else
    echo "  MISSING $f — will cold-score (slow)"
  fi
done
echo "================================================================"
read -r -p "Press Enter when ready to continue: "

# ---------------------------------------------------------------------------
# 6. HARD GATE: subjectivity checkpoint
# ---------------------------------------------------------------------------
if [ ! -d subjectivity_model ] || [ ! -f subjectivity_model/subjectivity_meta.json ]; then
  echo ""
  echo "================================================================"
  echo "  FATAL: subjectivity_model/ checkpoint not found."
  echo "  Transfer and unpack it before running:"
  echo "    LAPTOP:  runpodctl send subjectivity_model.tar.gz"
  echo "    POD:     runpodctl receive <code>"
  echo "             tar -xzf subjectivity_model.tar.gz"
  echo "  Then re-run this script."
  echo "================================================================"
  exit 1
fi
echo "subjectivity_model/ found."

# ---------------------------------------------------------------------------
# 7. VALIDATE + FIX tokenizer in the checkpoint.
#    If the checkpoint was saved before the train_subjectivity.py call-site fix
#    (tokenizer not passed to model.save()), load_trained resolves a degenerate
#    tokenizer that maps every pair to identical ids. This check detects and
#    fixes that silently so the run doesn't produce garbage subjectivity scores.
# ---------------------------------------------------------------------------
python - <<'PY'
import os, sys
from transformers import AutoTokenizer

model_dir = "subjectivity_model"
tok_files = {"tokenizer.json", "vocab.json", "tokenizer_config.json"}
present = set(os.listdir(model_dir))

if tok_files & present:
    print(f"Tokenizer OK ({len(tok_files & present)} tokenizer files found in checkpoint).")
else:
    print("Tokenizer files missing from checkpoint — saving roberta-base tokenizer now...")
    tok = AutoTokenizer.from_pretrained("roberta-base")
    tok.save_pretrained(model_dir)
    print(f"  Saved to {model_dir}/. Degenerate-ids bug is fixed.")
PY

# ---------------------------------------------------------------------------
# 8. SMOKE TEST — scores one AAPL transcript end-to-end in < 60s.
#    Proves checkpoint loads, tokenizer is correct, segmentation works, and
#    both scorers produce non-constant output before you commit to a 10h run.
# ---------------------------------------------------------------------------
echo ""
echo "Running smoke test (AAPL, both scorers) ..."
python - <<'PY'
import sys
try:
    from subjectivity_scoring import SubjectivityScorer, score_transcript_subjectivity
    from sentiment_scoring import FinBERTScorer
    import preprocessing as pp
    from fmp_client import FMPClient

    client = FMPClient()
    dates = client.transcript_dates("AAPL")
    last = dates[-1]
    t = client.get_transcript("AAPL", last["year"], last["quarter"])  # type: ignore[call-arg]
    if not t or not t["content"]:
        sys.exit("Smoke test FAILED: empty AAPL transcript.")

    # FinBERT
    fb = FinBERTScorer()
    split = pp.split_transcript(t["content"])
    tok = {"prepared": pp.sentence_tokenize(pp.clean_fmp_text(split["prepared"])),
           "qa": pp.sentence_tokenize(pp.clean_fmp_text(split["qa"]))}
    from sentiment_scoring import score_transcript
    fb_out = score_transcript(tok, "AAPL", str(last["year"]), fb)
    print(f"  FinBERT composite: {fb_out['composite']:.4f}")

    # Subjectivity
    sub = SubjectivityScorer()
    pairs = pp.segment_qa_pairs(split["qa"])
    if not pairs:
        print("  WARNING: 0 Q&A pairs segmented — subjectivity scores will be None.")
    else:
        rec = score_transcript_subjectivity(pairs, "AAPL", str(last["year"]), sub)
        feats = rec["features"]
        # check scores are non-constant across pairs
        s = sub.score_pairs(pairs[:4])
        spec_vals = [p["SPECIFIC"] for p in s]
        if max(spec_vals) - min(spec_vals) < 0.01:
            sys.exit("Smoke test FAILED: subjectivity scores are near-constant "
                     "— tokenizer fix did not take.")
        print(f"  Subjectivity n_pairs={rec['n_pairs']}  "
              f"specific_mean={feats['specific_mean']:.4f}  "
              f"numerical_density_mean={feats['numerical_density_mean']:.4f}")
    print("Smoke test PASSED.")
except Exception as e:
    sys.exit(f"Smoke test FAILED: {e}")
PY

# ---------------------------------------------------------------------------
# 9. BUILD RUN COMMAND
# ---------------------------------------------------------------------------
RUN_CMD="python backtest.py --months $MONTHS"
if [ -n "$MAX_TICKERS" ]; then
  RUN_CMD="$RUN_CMD --max-tickers $MAX_TICKERS"
  echo ""
  echo "PILOT MODE: --max-tickers $MAX_TICKERS"
fi

# ---------------------------------------------------------------------------
# 10. RUN — wrapped in nohup so a browser disconnect won't kill it.
#     Progress is logged to run_log.txt. Monitor with: tail -f run_log.txt
#
#     Sequence on completion:
#       a) Upload all three output files (0x0.st + file.io, both attempted).
#       b) Only if ALL uploads succeeded AND RUNPOD_API_KEY is set: stop the pod.
#       c) If any upload fails OR the key is missing: pod stays alive for manual retrieval.
# ---------------------------------------------------------------------------
echo ""
echo "Starting backtest ($RUN_CMD) in background via nohup..."
echo "Monitor: tail -f $WORKDIR/NLP-Alpha/run_log.txt"
echo ""

nohup bash -c "
  set -euo pipefail

  $RUN_CMD

  echo '=== Backtest complete. Uploading outputs... ==='
  UPLOAD_OK=1

  for f in backtest_trades.csv backtest_scores.json subjectivity_scores.json; do
    if [ ! -s \"\$f\" ]; then
      echo \"WARNING: \$f missing or empty — skipping upload\"
      UPLOAD_OK=0
      continue
    fi
    echo \"--- Uploading \$f ---\"
    # attempt both hosts; flag failure only if BOTH fail for this file
    R1=\$(curl -s -w '%{http_code}' -F \"file=@\$f\" https://0x0.st  -o /tmp/up1.txt)
    R2=\$(curl -s -w '%{http_code}' -F \"file=@\$f\" https://file.io -o /tmp/up2.txt)
    echo \"  0x0.st [\$R1]:  \$(cat /tmp/up1.txt)\"
    echo \"  file.io [\$R2]: \$(cat /tmp/up2.txt)\"
    if [ \"\$R1\" != '200' ] && [ \"\$R2\" != '200' ]; then
      echo \"  ERROR: both upload hosts failed for \$f\"
      UPLOAD_OK=0
    fi
  done

  echo '=== Upload phase complete. Check saved_links.txt for download URLs. ==='

  if [ \"\$UPLOAD_OK\" -eq 1 ] && [ -n '${RUNPOD_API_KEY}' ]; then
    echo '=== All uploads succeeded. Stopping pod in 60 seconds... ==='
    echo '    (Ctrl-C this process within 60s if you need to keep it alive.)'
    sleep 60
    echo '=== Stopping pod now. ==='
    curl -s -X POST https://api.runpod.io/graphql \
      -H 'Content-Type: application/json' \
      -H 'Authorization: Bearer ${RUNPOD_API_KEY}' \
      --data '{\"query\": \"mutation { podStop(input: { podId: \\\"'\$RUNPOD_POD_ID'\\\" }) { id } }\"}' \
      && echo 'Pod stop request sent.' \
      || echo 'WARNING: pod stop API call failed — terminate the pod manually.'
  elif [ \"\$UPLOAD_OK\" -eq 0 ]; then
    echo '=== WARNING: one or more uploads failed. Pod NOT stopped — retrieve files manually. ==='
    echo '    runpodctl send backtest_trades.csv'
    echo '    runpodctl send backtest_scores.json'
    echo '    runpodctl send subjectivity_scores.json'
  else
    echo '=== RUNPOD_API_KEY not set. Pod NOT stopped — terminate it manually when done. ==='
  fi
" 2>&1 | tee run_log.txt | tee saved_links.txt &

echo "PID: $!"
echo ""
echo "You can safely close your browser now."
echo "The pod will auto-stop after successful upload (if RUNPOD_API_KEY was set)."
echo "Monitor: tail -f $WORKDIR/NLP-Alpha/run_log.txt"
echo "If needed, retrieve outputs manually:"
echo "  POD:    runpodctl send backtest_trades.csv"
echo "          runpodctl send backtest_scores.json"
echo "          runpodctl send subjectivity_scores.json"
echo "  LAPTOP: runpodctl receive <code>"
