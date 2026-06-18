"""
Script used to fine-tune a multi-task PLM (RoBERTa) on SubjECTive-QA.

Produces a checkpoint with 6 independent 3-class classification heads (Assertive, Cautious,
Optimistic, Specific, Clear, Relevant; labels 0/1/2 = negatively / neutral / positively
demonstrative). Phase 3 loads this checkpoint and runs inference over the QA-pair corpus.

Design notes:
  * Plain PyTorch loop (no HuggingFace Trainer). Only stable primitives — AutoModel,
    AutoTokenizer, torch — so it doesn't depend on Trainer/TrainingArguments kwargs that
    churn between transformers versions.
  * bf16 autocast on capable GPUs, else fp32. No GradScaler.
  * Per-head accuracy + macro-F1 reported every epoch and on the held-out test split.
  * Inverse-frequency class weights (on by default) so heads don't collapse to the
    majority "neutral" class on imbalanced dimensions. Disable with --no-class-weights.

SCOPE NOTE (logged into the checkpoint meta): trained on SubjECTive-QA only, 120
large-cap NYSE companies, 2007-2021, QA pairs only, NO in-domain relabel. Transfer to a
broader / smaller-cap universe is UNVALIDATED. If a dimension's feature looks dead
downstream, suspect transfer before concluding the dimension is uninformative.
"""

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from datasets import load_dataset


# --- dataset constants ---

DATASET_ID = "gtfintechlab/SubjECTive-QA"
QUESTION_COL = "QUESTION"
ANSWER_COL = "ANSWER"
DIMENSIONS = ["ASSERTIVE", "CAUTIOUS", "OPTIMISTIC", "SPECIFIC", "CLEAR", "RELEVANT"]
N_CLASSES = 3
META_FILENAME = "subjectivity_meta.json"
HEADS_FILENAME = "heads.pt"

SCOPE_NOTE = (
    "Trained on SubjECTive-QA only (120 large-cap NYSE companies, 2007-2021, QA pairs "
    "only; no in-domain relabel). Large-cap -> broader-universe transfer is UNVALIDATED. "
    "If a dimension looks dead downstream, suspect transfer before the dimension itself."
)


# --- model ---

class ClassificationHead(nn.Module):
    """RoBERTa-style classification head on the <s> token representation."""
    def __init__(self, hidden: int, n_classes: int = N_CLASSES, dropout: float = 0.1):
        super().__init__()
        self.dense = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, n_classes)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor: # pooled: (B, hidden)
        x = self.dropout(pooled)
        x = torch.tanh(self.dense(x))
        x = self.dropout(x)
        return self.out(x) # (B, n_classes)


class MultiTaskSubjectivityModel(nn.Module):
    """One shared encoder, six independent heads. forward() returns {dim -> (B, 3) logits}."""
    def __init__(self, encoder, dimensions, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.dimensions = list(dimensions)
        h = encoder.config.hidden_size # stores length of internal RoBERTa vectors
        # creates 6 ClassificationHead instances
        self.heads = nn.ModuleDict(
            {d: ClassificationHead(h, N_CLASSES, dropout) for d in self.dimensions}
        )

    def forward(self, input_ids, attention_mask) -> dict:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0] # <s> token == [CLS]
        # does a forward pass on each head and returns them in a dict
        return {d: self.heads[d](pooled) for d in self.dimensions}

    # --- constructors ---

    @classmethod
    def new(cls, base_model: str, dimensions, dropout: float = 0.1):
        """Fresh model: download the base encoder, attach untrained heads."""
        return cls(AutoModel.from_pretrained(base_model), dimensions, dropout)

    @classmethod
    def load_trained(cls, out_dir: str, device: str = "cpu"):
        """Reload a saved checkpoint for inference. Returns (model, tokenizer, meta)."""
        with open(os.path.join(out_dir, META_FILENAME)) as f:
            meta = json.load(f) # reads metadata json to get dims and dropout for rebuild
        encoder = AutoModel.from_pretrained(out_dir)
        model = cls(encoder, meta["dimensions"], meta.get("dropout", 0.1))
        # loads the saved heads state dict
        state = torch.load(os.path.join(out_dir, HEADS_FILENAME), map_location=device)
        model.heads.load_state_dict(state) # populates heads with saved weights
        model.to(device).eval() # sets inference mode (disables dropout)
        tokenizer = AutoTokenizer.from_pretrained(out_dir)
        return model, tokenizer, meta

    def save(self, out_dir, meta, tokenizer=None):
        os.makedirs(out_dir, exist_ok=True)
        self.encoder.save_pretrained(out_dir)
        if tokenizer is not None:
            tokenizer.save_pretrained(out_dir)
        torch.save(self.heads.state_dict(), os.path.join(out_dir, HEADS_FILENAME))
        with open(os.path.join(out_dir, META_FILENAME), "w") as f:
            json.dump(meta, f, indent=2)


# --- data ---

class QADataset(Dataset):
    """Wraps one HF split. __getitem__ -> (question, answer, label_vector[6])."""
    def __init__(self, split, dimensions):
        self.q = split[QUESTION_COL]
        self.a = split[ANSWER_COL]
        # (N, 6) int matrix, column order == DIMENSIONS
        self.labels = np.stack([np.asarray(split[d], dtype=np.int64) for d in dimensions], axis=1)

    def __len__(self):
        return len(self.q)

    def __getitem__(self, i):
        return str(self.q[i] or ""), str(self.a[i] or ""), self.labels[i]


def make_collate(tokenizer, max_length: int):
    """
    Factory function.
    Tokenize each batch as a (question, answer) text-pair with dynamic padding.

    RoBERTa joins the pair as <s> question </s></s> answer </s> automatically.
    truncation=True trims the longer of the two when over budget.
    """
    def collate(batch):
        qs = [b[0] for b in batch]
        ans = [b[1] for b in batch]
        labels = torch.from_numpy(np.stack([b[2] for b in batch]))      # (B, 6) long
        enc = tokenizer(qs, ans, truncation=True, max_length=max_length,
                        padding=True, return_tensors="pt")
        return enc["input_ids"], enc["attention_mask"], labels
    return collate


def class_weights_from(train_split, dimensions, device):
    """Inverse-frequency 3-class weights per dimension, normalized so the mean weight ~= 1.
    Counters the heavy 'neutral' (label 1) skew so heads don't collapse to the majority class."""
    weights = {}
    for d in dimensions:
        # bincount counts occurrences of each int in the column
        counts = np.bincount(np.asarray(train_split[d], dtype=np.int64), minlength=N_CLASSES).astype(float)
        counts[counts == 0] = 1.0 # avoid div-by-zero on absent classes
        w = counts.sum() / (N_CLASSES * counts) # inverse frequency
        weights[d] = torch.tensor(w, dtype=torch.float, device=device)
    return weights


# --- loss & metrics ---

def compute_loss(logits, labels, dimensions, class_weights=None):
    """Sum of the six per-head cross-entropies. labels: (B, 6) long."""
    total = 0.0
    for i, d in enumerate(dimensions):
        w = class_weights[d] if class_weights else None
        total = total + F.cross_entropy(logits[d], labels[:, i], weight=w)
    return total


def macro_f1(preds: np.ndarray, golds: np.ndarray) -> float:
    """
    Unweighted mean F1 across the 3 classes (robust to label imbalance).
    F1 used to evaluate model balance by measuring precision and recall.
    Low F1 => trade-off b/w precision and recall.
    High F1 => reliable, balanced model.
    """
    f1s = []
    for c in range(N_CLASSES):
        tp = int(((preds == c) & (golds == c)).sum()) # true positives
        fp = int(((preds == c) & (golds != c)).sum()) # false positives
        fn = int(((preds != c) & (golds == c)).sum()) # false negatives
        prec = tp / (tp + fp) if (tp + fp) else 0.0 # precision
        rec = tp / (tp + fn) if (tp + fn) else 0.0 # recall
        # F1 formula
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return sum(f1s) / len(f1s)


@torch.inference_mode() # decorator to disable gradient tracking during inference
def evaluate(model, loader, dimensions, device, amp_dtype, autocast_enabled):
    """Per-head accuracy and macro-F1 over a split. Returns {dim: {acc, f1}} plus mean_f1."""
    model.eval()
    preds = {d: [] for d in dimensions}
    golds = {d: [] for d in dimensions}
    for input_ids, attention_mask, labels in loader:
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
            logits = model(input_ids, attention_mask)
        for i, d in enumerate(dimensions):
            preds[d].append(logits[d].float().argmax(-1).cpu().numpy())
            golds[d].append(labels[:, i].numpy())
    out = {}
    for d in dimensions:
        # concatenates all batch predictions and golds into full-split arrays
        p = np.concatenate(preds[d]); g = np.concatenate(golds[d])
        out[d] = {"acc": float((p == g).mean()), "f1": macro_f1(p, g)}
    mean_f1 = float(np.mean([out[d]["f1"] for d in dimensions]))
    return out, mean_f1


def print_table(title, metrics, mean_f1):
    print(f"\n  {title}")
    # formats a fixed-width table
    print(f"    {'dimension':<12} {'acc':>6} {'macroF1':>8}")
    for d in DIMENSIONS:
        print(f"    {d:<12} {metrics[d]['acc']:>6.3f} {metrics[d]['f1']:>8.3f}")
    print(f"    {'MEAN':<12} {'':>6} {mean_f1:>8.3f}")


# --- training ---

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # bf16 preferred over fp16 (better dynamic range, no GradScaler needed)
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float32
    autocast_enabled = use_bf16
    print(f"Device: {device} | precision: {'bf16' if use_bf16 else 'fp32'}")
    if device.type == "cpu":
        print("  (CPU detected — fine for a smoke test, slow for the real run.)")

    print(f"Loading {DATASET_ID} config '{args.dataset_config}' ...")
    ds = load_dataset(DATASET_ID, args.dataset_config)
    # handles both HF dataset naming conventions
    val_key = "val" if "val" in ds else ("validation" if "validation" in ds else None)
    if val_key is None:
        raise SystemExit(f"No validation split found. Splits present: {list(ds.keys())}")
    # fail fast if the schema ever drifts
    missing = [c for c in [QUESTION_COL, ANSWER_COL, *DIMENSIONS] if c not in ds["train"].column_names]
    if missing:
        raise SystemExit(f"Dataset is missing expected columns: {missing}")
    if args.smoke_test:
        print("\n  *** SMOKE-TEST MODE: slicing to 64 train / 32 val / 32 test, 1 epoch ***")
        args.epochs = 1
        ds["train"] = ds["train"].select(range(min(64, len(ds["train"]))))
        ds[val_key] = ds[val_key].select(range(min(32, len(ds[val_key]))))
        ds["test"]  = ds["test"].select(range(min(32, len(ds["test"]))))
    print(f"  splits: train={len(ds['train'])}  {val_key}={len(ds[val_key])}  test={len(ds['test'])}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    collate = make_collate(tokenizer, args.max_length)
    # shuffles examples over every epoch so order isn't learned
    train_loader = DataLoader(QADataset(ds["train"], DIMENSIONS), batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate)
    val_loader = DataLoader(QADataset(ds[val_key], DIMENSIONS), batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate)
    test_loader = DataLoader(QADataset(ds["test"], DIMENSIONS), batch_size=args.batch_size,
                             shuffle=False, collate_fn=collate)

    class_weights = None if args.no_class_weights else class_weights_from(ds["train"], DIMENSIONS, device)
    if class_weights:
        print("  using inverse-frequency class weights per head")

    model = MultiTaskSubjectivityModel.new(args.base_model, DIMENSIONS, args.dropout).to(device)

    # AdamW applies weight decay separate to gradient update (best for fine-tuning)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    # LR linearly increases from 0 to args.lr over the first warmup_ratio * total_steps steps
    # then linearly decays back to 0.
    # default warmup is 6% of total steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(args.warmup_ratio * total_steps), total_steps)

    best_f1, best_val_metrics = -1.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for input_ids, attention_mask, labels in train_loader:
            input_ids, attention_mask, labels = (
                input_ids.to(device), attention_mask.to(device), labels.to(device))
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                logits = model(input_ids, attention_mask)
                loss = compute_loss(logits, labels, DIMENSIONS, class_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += loss.item()

        val_metrics, val_f1 = evaluate(model, val_loader, DIMENSIONS, device, amp_dtype, autocast_enabled)
        print(f"\nEpoch {epoch}/{args.epochs}  train_loss={running / len(train_loader):.4f}")
        print_table("validation", val_metrics, val_f1)

        if val_f1 > best_f1: # checkpoint the best epoch by mean val macro-F1
            best_f1, best_val_metrics = val_f1, val_metrics
            meta = {
                "base_model": args.base_model, "dimensions": DIMENSIONS,
                "n_classes": N_CLASSES, "max_length": args.max_length,
                "dropout": args.dropout, "dataset_config": args.dataset_config,
                "label_scheme": {"0": "negatively demonstrative", "1": "neutral",
                                 "2": "positively demonstrative"},
                "best_epoch": epoch, "val_mean_f1": best_f1,
                "val_metrics": best_val_metrics, "scope_note": SCOPE_NOTE,
            }
            model.save(args.out_dir, meta, tokenizer)
            print(f"  ** new best (mean val F1 {best_f1:.3f}) — checkpoint -> {args.out_dir}/")

    # final: reload the best checkpoint and report the held-out TEST table
    print("\n" + "=" * 60)
    print(f"Best mean val F1: {best_f1:.3f}. Evaluating best checkpoint on TEST split.")
    best_model, _, meta = MultiTaskSubjectivityModel.load_trained(args.out_dir, device=str(device))
    test_metrics, test_f1 = evaluate(best_model, test_loader, DIMENSIONS, device, amp_dtype, autocast_enabled)
    print_table("TEST (held-out)", test_metrics, test_f1)

    meta["test_mean_f1"] = test_f1
    meta["test_metrics"] = test_metrics
    with open(os.path.join(args.out_dir, META_FILENAME), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nCheckpoint + metrics -> {args.out_dir}/")
    print(f"SCOPE NOTE: {SCOPE_NOTE}")


def main():
    ap = argparse.ArgumentParser(description="Fine-tune a multi-task PLM on SubjECTive-QA.")
    ap.add_argument("--dataset-config", default="5768",
                    help="SubjECTive-QA config = split seed (e.g. 5768). Keep it fixed across runs.")
    ap.add_argument("--base-model", default="roberta-base")
    ap.add_argument("--out-dir", default="subjectivity_model")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--warmup-ratio", type=float, default=0.06)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42, help="Training seed (separate from the dataset config seed).")
    ap.add_argument("--no-class-weights", action="store_true",
                    help="Disable inverse-frequency class weighting.")
    ap.add_argument("--smoke-test", action="store_true",
                    help="Slice to 64/32/32 examples and 1 epoch. Proves auth/download/"
                         "train/save/tar end-to-end in ~2 min before the real run.")
    run(ap.parse_args())


if __name__ == "__main__":
    main()