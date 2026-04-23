#!/usr/bin/env bash
# =============================================================================
# run_batch_sweep.sh — Run the GPU batch-size sweep and plot the results.
#
# Sweeps batch_size ∈ {1, 2, 4, 8} for both colocated and disaggregated designs
# on real GPU hardware, then renders comparison plots.
#
# Usage:
#   bash run_batch_sweep.sh                              # full sweep (2 GPUs)
#   bash run_batch_sweep.sh --baseline-only              # colocated only (1 GPU)
#   bash run_batch_sweep.sh --num-requests 64            # more requests per run
#   bash run_batch_sweep.sh --batch-sizes 1,2,4,8,16     # custom sweep
#   bash run_batch_sweep.sh --model MODEL
#   bash run_batch_sweep.sh --prefill-gpu 0 --decode-gpu 1
#   bash run_batch_sweep.sh --skip-plots                 # sweep only, no plots
# =============================================================================

set -euo pipefail

MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PREFILL_GPU=0
DECODE_GPU=1
BASELINE_GPU=0
NUM_REQUESTS=32
BATCH_SIZES="1,2,4,8"
ARRIVAL_RATE=0.0
OUTPUT_LOW=8
OUTPUT_HIGH=32
MAX_PROMPT_TOKENS=512
TTFT_SLO=2.0
TPOT_SLO=0.05
BASELINE_ONLY=""
SKIP_PLOTS=false
SHAREGPT_JSONL=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)             MODEL="$2";             shift 2 ;;
    --prefill-gpu)       PREFILL_GPU="$2";       shift 2 ;;
    --decode-gpu)        DECODE_GPU="$2";        shift 2 ;;
    --baseline-gpu)      BASELINE_GPU="$2";      shift 2 ;;
    --num-requests)      NUM_REQUESTS="$2";      shift 2 ;;
    --batch-sizes)       BATCH_SIZES="$2";       shift 2 ;;
    --arrival-rate)      ARRIVAL_RATE="$2";      shift 2 ;;
    --output-low)        OUTPUT_LOW="$2";        shift 2 ;;
    --output-high)       OUTPUT_HIGH="$2";       shift 2 ;;
    --max-prompt-tokens) MAX_PROMPT_TOKENS="$2"; shift 2 ;;
    --ttft-slo)          TTFT_SLO="$2";          shift 2 ;;
    --tpot-slo)          TPOT_SLO="$2";          shift 2 ;;
    --baseline-only)     BASELINE_ONLY="--baseline-only"; shift ;;
    --skip-plots)        SKIP_PLOTS=true;        shift ;;
    --sharegpt-jsonl)    SHAREGPT_JSONL="$2";    shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

JSONL_FLAG=""
if [[ -n "$SHAREGPT_JSONL" ]]; then
  JSONL_FLAG="--sharegpt-jsonl $SHAREGPT_JSONL"
fi

echo "======================================================================"
echo "  GPU Batch-Size Sweep"
echo "  Model:         $MODEL"
echo "  Batch sizes:   $BATCH_SIZES"
echo "  Num requests:  $NUM_REQUESTS"
echo "  Output tokens: [$OUTPUT_LOW, $OUTPUT_HIGH]"
echo "  Arrival rate:  $ARRIVAL_RATE req/s"
echo "  Prefill GPU:   $PREFILL_GPU"
echo "  Decode GPU:    $DECODE_GPU"
echo "  Baseline only: ${BASELINE_ONLY:-no}"
echo "======================================================================"
echo ""

echo "Python:  $(command -v python)"
echo "Version: $(python --version 2>&1)"
echo ""

# Probe torch + CUDA and show any import / driver error.
# Temporarily disable errexit so we can capture probe failures without aborting.
set +e
PROBE_OUTPUT=$(python - <<'PY' 2>&1
import sys
try:
    import torch
except Exception as e:
    print(f"TORCH_IMPORT_ERROR: {type(e).__name__}: {e}")
    sys.exit(2)
if not torch.cuda.is_available():
    print(f"CUDA_NOT_AVAILABLE: torch={torch.__version__}")
    sys.exit(3)
for i in range(torch.cuda.device_count()):
    print(f"cuda:{i} {torch.cuda.get_device_name(i)}")
print(f"NUM_GPUS={torch.cuda.device_count()}")
PY
)
PROBE_STATUS=$?
set -e

if [[ $PROBE_STATUS -ne 0 ]]; then
  echo "ERROR during torch/CUDA probe (exit $PROBE_STATUS):" >&2
  echo "$PROBE_OUTPUT" >&2
  echo "" >&2
  echo "Common fixes:" >&2
  echo "  * pip install -r requirements.txt    (if torch is missing)" >&2
  echo "  * Activate the env that has torch+CUDA (conda/venv)" >&2
  echo "  * Check 'nvidia-smi' works from this shell" >&2
  exit 1
fi

NUM_GPUS=$(echo "$PROBE_OUTPUT" | sed -n 's/^NUM_GPUS=//p')
echo "Detected $NUM_GPUS GPU(s):"
echo "$PROBE_OUTPUT" | grep -E '^cuda:' || true

if [[ "$NUM_GPUS" -lt 2 && -z "$BASELINE_ONLY" ]]; then
  echo "Only 1 GPU visible; forcing --baseline-only."
  BASELINE_ONLY="--baseline-only"
fi

echo ""
echo ">>> Running GPU batch-size sweep..."
python -m src.experiments.run_batch_sweep_gpu \
  --model "$MODEL" \
  --num-requests "$NUM_REQUESTS" \
  --batch-sizes "$BATCH_SIZES" \
  --baseline-gpu "$BASELINE_GPU" \
  --prefill-gpu "$PREFILL_GPU" \
  --decode-gpu "$DECODE_GPU" \
  --arrival-rate "$ARRIVAL_RATE" \
  --output-low "$OUTPUT_LOW" \
  --output-high "$OUTPUT_HIGH" \
  --max-prompt-tokens "$MAX_PROMPT_TOKENS" \
  --ttft-slo "$TTFT_SLO" \
  --tpot-slo "$TPOT_SLO" \
  $BASELINE_ONLY \
  $JSONL_FLAG

CSV_PATH="results/sweep_type_1/gpu_batch_sweep.csv"
if [[ ! -f "$CSV_PATH" ]]; then
  echo "ERROR: expected $CSV_PATH but it was not produced." >&2
  exit 1
fi

if [[ "$SKIP_PLOTS" == "true" ]]; then
  echo ""
  echo "Plotting skipped (--skip-plots). CSV at $CSV_PATH."
  exit 0
fi

echo ""
echo ">>> Generating plots from $CSV_PATH ..."
python -m src.experiments.plot_gpu_batch_sweep \
  --csv "$CSV_PATH" \
  --out-dir "results/sweep_type_1/plots"

echo ""
echo "======================================================================"
echo "  Done."
echo "  CSV:    $CSV_PATH"
echo "  Plots:  results/sweep_type_1/plots/"
echo "======================================================================"
