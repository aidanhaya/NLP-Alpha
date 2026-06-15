#!/usr/bin/env bash
# runpod_setup.sh — provision a RunPod GPU pod to run backtest.py on CUDA.
#
# Use a RunPod template that ships a CUDA-enabled PyTorch (e.g. the official
# "RunPod PyTorch 2.x" image). Open the pod's web terminal and run:
#   bash runpod_setup.sh
#
# Key safeguards:
#   1) Hard-fails if CUDA is unavailable (no silent CPU runs).
#   2) Pauses before running so you can seed the score cache via runpodctl.
#   3) Auto-uploads both output files on completion — no manual retrieval needed.
set -euo pipefail

# ---------------------------------------------------------------------------
# 0. CONFIG
# ---------------------------------------------------------------------------
REPO="https://github.com/aidanhaya/NLP-Alpha.git"
export FMP_API_KEY="${FMP_API_KEY:-ivpch0qtpEYBQ7Ojg35eK3ed9LiJczCj}"
MONTHS=36

WORKDIR=/workspace
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
# 2. Dependencies (skip torch/playwright/ib-insync — not needed for backtest)
# ---------------------------------------------------------------------------
grep -v -iE '^(torch|playwright|ib-insync|matplotlib|seaborn)([=<>! ]|$)' \
    requirements.txt > /tmp/reqs.txt
pip install -q -r /tmp/reqs.txt
python -c "import nltk; nltk.download('punkt_tab', quiet=True)"

# ---------------------------------------------------------------------------
# 3. HARD GATE: refuse to run on CPU
# ---------------------------------------------------------------------------
python - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    sys.exit("FATAL: torch.cuda.is_available() == False.\n"
             "You'd be paying GPU prices to run FinBERT on CPU.\n"
             "Fix: use a RunPod PyTorch/CUDA template, then re-run.")
print(f"CUDA OK -> {torch.cuda.get_device_name(0)} | torch {torch.__version__}")
PY

# ---------------------------------------------------------------------------
# 4. SEED THE CACHE — pause here so you can transfer backtest_scores.json.
#    On your LAPTOP:  runpodctl send backtest_scores.json
#    On THIS pod:     runpodctl receive <code>
#    Then press Enter below to continue.
# ---------------------------------------------------------------------------
if [ -f backtest_scores.json ]; then
  python -c "import json; print('Score cache present:', len(json.load(open('backtest_scores.json'))), 'transcripts')"
else
  echo ""
  echo "================================================================"
  echo "  NO score cache found — full re-score will take ~13 hours."
  echo "  To seed the cache:"
  echo "    LAPTOP:  runpodctl send backtest_scores.json"
  echo "    POD:     runpodctl receive <code>"
  echo "  Then press Enter to continue, or Ctrl-C to abort."
  echo "================================================================"
  read -r -p "Press Enter when ready (or Enter to skip and full re-score): "
fi

# ---------------------------------------------------------------------------
# 5. RUN — wrapped in nohup so a browser disconnect won't kill it.
#    Progress is logged to run_log.txt. Monitor with: tail -f run_log.txt
# ---------------------------------------------------------------------------
echo ""
echo "Starting backtest (--months $MONTHS) in background via nohup..."
echo "Monitor progress: tail -f /workspace/NLP-Alpha/run_log.txt"
echo ""

nohup bash -c "
  python backtest.py --months $MONTHS
  echo '=== Backtest complete. Uploading outputs... ==='
  for f in backtest_trades.csv backtest_scores.json; do
    if [ -s \"\$f\" ]; then
      echo \"--- Uploading \$f ---\"
      echo \"0x0.st:\" && curl -s -F \"file=@\$f\" https://0x0.st && echo
      echo \"file.io:\" && curl -s -F \"file=@\$f\" https://file.io && echo
    else
      echo \"WARNING: \$f missing or empty — skipping upload\"
    fi
  done
  echo '=== All uploads done. Check saved_links.txt ==='
" 2>&1 | tee run_log.txt | tee saved_links.txt &

echo "PID: $!"
echo ""
echo "You can safely close your browser now."
echo "When done, check: cat /workspace/NLP-Alpha/saved_links.txt"