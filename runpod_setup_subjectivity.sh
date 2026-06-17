#!/usr/bin/env bash
# runpod_setup_subjectivity.sh — provision a RunPod GPU pod to fine-tune the multi-task
# subjectivity model (Phase 2) with train_subjectivity.py on CUDA.
#
# Use a RunPod template that ships CUDA-enabled PyTorch (e.g. "RunPod PyTorch 2.x").
# Open the pod's web terminal and run:
#   export HF_TOKEN=hf_...        # account must have ACCEPTED the dataset terms first
#   bash runpod_setup_subjectivity.sh
#
# Safeguards:
#   1) Hard-fails if CUDA is unavailable (no silent CPU runs).
#   2) Hard-fails if HF_TOKEN is missing (the dataset is gated).
#   3) Auto-packages the checkpoint on completion for runpodctl retrieval.
set -euo pipefail

# ---------------------------------------------------------------------------
# 0. CONFIG
# ---------------------------------------------------------------------------
REPO="https://github.com/aidanhaya/NLP-Alpha.git"
WORKDIR=/workspace
DATASET_CONFIG="${DATASET_CONFIG:-5768}"   # SubjECTive-QA split seed; keep fixed across runs
OUT_DIR="${OUT_DIR:-subjectivity_model}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-16}"

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
# 2. Dependencies — training needs only transformers + datasets + hub.
#    torch comes from the CUDA template; do NOT reinstall it (would risk a CPU build).
# ---------------------------------------------------------------------------
pip install -q "transformers" "datasets" "huggingface_hub"

# ---------------------------------------------------------------------------
# 3. HARD GATE: refuse to run on CPU
# ---------------------------------------------------------------------------
python - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    sys.exit("FATAL: torch.cuda.is_available() == False.\n"
             "Use a RunPod PyTorch/CUDA template, then re-run.")
bf16 = torch.cuda.is_bf16_supported()
print(f"CUDA OK -> {torch.cuda.get_device_name(0)} | torch {torch.__version__} | bf16={bf16}")
PY

# ---------------------------------------------------------------------------
# 4. HARD GATE: Hugging Face auth (SubjECTive-QA is gated)
# ---------------------------------------------------------------------------
if [ -z "${HF_TOKEN:-}" ]; then
  echo ""
  echo "================================================================"
  echo "  FATAL: HF_TOKEN is not set, but the dataset is gated."
  echo "    1) Accept terms: https://huggingface.co/datasets/gtfintechlab/SubjECTive-QA"
  echo "    2) Make a read token: https://huggingface.co/settings/tokens"
  echo "    3) export HF_TOKEN=hf_xxx   and re-run this script."
  echo "================================================================"
  exit 1
fi
export HF_TOKEN
# Persist for hub/datasets. Newer CLI is 'hf'; fall back to legacy 'huggingface-cli'.
hf auth login --token "$HF_TOKEN" 2>/dev/null \
  || huggingface-cli login --token "$HF_TOKEN" 2>/dev/null \
  || echo "  (CLI login skipped; datasets will use HF_TOKEN from the environment.)"
python -c "from huggingface_hub import whoami; print('HF auth OK ->', whoami()['name'])"

# ---------------------------------------------------------------------------
# 5. SMOKE TEST — runs in ~2 min; proves auth/download/train/save/tar before
#    committing to the full run. Hard-fails if anything in the pipeline is broken.
# ---------------------------------------------------------------------------
echo ""
echo "Running smoke test (64 examples, 1 epoch) ..."
python train_subjectivity.py \
  --dataset-config "$DATASET_CONFIG" \
  --out-dir "${OUT_DIR}_smoke" \
  --smoke-test
echo ""
echo "================================================================"
echo "  Smoke test passed. Full run will take minutes-to-an-hour."
echo "  Press Enter to continue, or Ctrl-C to abort."
echo "================================================================"
read -r -p ""

# ---------------------------------------------------------------------------
# 6. TRAIN — wrapped in nohup so a browser disconnect won't kill it.
#    Fine-tuning ~2.7k pairs is fast (minutes-to-an-hour), but the dataset download
#    and first run benefit from being disconnect-proof. Monitor with: tail -f run_log.txt

# ---------------------------------------------------------------------------
echo ""
echo "Starting fine-tune (config $DATASET_CONFIG, $EPOCHS epochs) via nohup..."
echo "Monitor: tail -f $WORKDIR/NLP-Alpha/run_log.txt"
echo ""

nohup bash -c "
  python train_subjectivity.py \
    --dataset-config '$DATASET_CONFIG' \
    --epochs '$EPOCHS' \
    --batch-size '$BATCH_SIZE' \
    --out-dir '$OUT_DIR'
  echo '=== Training complete. Packaging checkpoint... ==='
  if [ -d '$OUT_DIR' ]; then
    tar -czf subjectivity_model.tar.gz '$OUT_DIR'
    echo \"Created subjectivity_model.tar.gz (\$(du -h subjectivity_model.tar.gz | cut -f1))\"
    echo '--- Retrieve it on your laptop with runpodctl ---'
    echo '  POD:    runpodctl send subjectivity_model.tar.gz'
    echo '  LAPTOP: runpodctl receive <code>'
  else
    echo \"WARNING: '$OUT_DIR' missing — training may have failed. Check run_log.txt\"
  fi
  echo '=== Done. ==='
" 2>&1 | tee run_log.txt &

echo "PID: $!"
echo ""
echo "You can safely close your browser now."
echo "When done: the run_log.txt tail prints the runpodctl send code for the checkpoint."