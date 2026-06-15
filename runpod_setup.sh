#!/usr/bin/env bash
# runpod_setup.sh — provision a RunPod *GPU* pod to run backtest.py on CUDA.
#
# Use a RunPod template that ships a CUDA-enabled PyTorch (e.g. the official
# "RunPod PyTorch 2.x" image). Open the pod's web terminal and paste this, OR
# upload the file and run:  bash runpod_setup.sh
#
# The two things people get wrong — and that this script guards against:
#   1) silently running on CPU because torch has no CUDA  -> hard-fails below;
#   2) re-scoring 160k transcripts from scratch because the score cache wasn't
#      carried over -> see the "SEED THE CACHE" step.
set -euo pipefail

# ---------------------------------------------------------------------------
# 0. CONFIG — edit these two lines (or set FMP_API_KEY in the pod's env vars UI)
# ---------------------------------------------------------------------------
REPO="https://github.com/aidanhaya/NLP-Alpha.git"
export FMP_API_KEY="${FMP_API_KEY:-ivpch0qtpEYBQ7Ojg35eK3ed9LiJczCj}"

WORKDIR=/workspace                     # persisted on RunPod network volumes
cd "$WORKDIR"

# ---------------------------------------------------------------------------
# 1. Get the code
# ---------------------------------------------------------------------------
if [ -d NLP-Alpha/.git ]; then
  cd NLP-Alpha && git pull --ff-only
else
  git clone "$REPO" && cd NLP-Alpha
fi

# ---------------------------------------------------------------------------
# 2. Dependencies.
#    The PyTorch template already ships a CUDA torch — do NOT reinstall it from
#    PyPI (risks pulling a mismatched/CPU wheel). backtest.py also doesn't use
#    playwright or ib-insync, so skip those (playwright pulls browser binaries).
# ---------------------------------------------------------------------------
grep -v -iE '^(torch|playwright|ib-insync|matplotlib|seaborn)([=<>! ]|$)' \
    requirements.txt > /tmp/reqs.txt
pip install -q -r /tmp/reqs.txt
python -c "import nltk; nltk.download('punkt_tab', quiet=True)"   # sentence tokenizer

# ---------------------------------------------------------------------------
# 3. HARD GATE: refuse to run on CPU. Renting a GPU and running on CPU is the
#    single most expensive mistake here — fail loudly instead of silently.
# ---------------------------------------------------------------------------
python - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    sys.exit("FATAL: torch.cuda.is_available() == False.\n"
             "You'd be paying GPU prices to run FinBERT on the pod's CPU.\n"
             "Fix: use a RunPod PyTorch/CUDA template, or install a CUDA torch\n"
             "build (e.g. the cu12x wheel matching this pod), then re-run.")
print(f"CUDA OK -> {torch.cuda.get_device_name(0)} | torch {torch.__version__}")
PY

# ---------------------------------------------------------------------------
# 4. SEED THE CACHE (optional but do it — saves re-scoring ~160k transcripts).
#    From your LAPTOP, in the repo dir:   runpodctl send backtest_scores.json
#    It prints a one-time code; on THIS pod run:  runpodctl receive <code>
#    Put the file at: $WORKDIR/NLP-Alpha/backtest_scores.json  (cwd of the run)
# ---------------------------------------------------------------------------
if [ -f backtest_scores.json ]; then
  python -c "import json;print('score cache present:',len(json.load(open('backtest_scores.json'))),'transcripts')"
else
  echo ">> NO score cache found. Either seed it (step 4) or accept a full re-score."
fi

# ---------------------------------------------------------------------------
# 5. RUN. GPU saturates fast on FinBERT — bump the batch size if VRAM allows by
#    editing FinBERTScorer(batch_size=...). --max-tickers N for a quick smoke test.
# ---------------------------------------------------------------------------
python backtest.py --months 12

# ---------------------------------------------------------------------------
# 6. RETRIEVE RESULTS before terminating the pod (ephemeral disk is wiped!).
#    On THIS pod:   runpodctl send backtest_trades.csv backtest_scores.json
#    On your LAPTOP: runpodctl receive <code>
# ---------------------------------------------------------------------------
echo
echo "DONE. Retrieve outputs, then terminate the pod:"
echo "  runpodctl send backtest_trades.csv backtest_scores.json"