#!/usr/bin/env bash
# =============================================================================
# run_workload_sweep.sh — 3D sweep (batch_size x num_requests x arrival_rate)
# on real GPUs, then render comparison plots.
#
# Usage:
#   bash run_workload_sweep.sh                                          # defaults
#   bash run_workload_sweep.sh --baseline-only                          # 1 GPU only
#   bash run_workload_sweep.sh --batch-sizes 1,2,4,8
#   bash run_workload_sweep.sh --num-requests 16,32,64,128
#   bash run_workload_sweep.sh --arrival-rates 0.0,1.0,2.0,4.0,8.0
#   bash run_workload_sweep.sh --out-dir results/milestone/workload_sweep_v2
#   bash run_workload_sweep.sh --skip-plots
# =============================================================================

set -euo pipefail

MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PREFILL_GPU=0
DECODE_GPU=1
BASELINE_GPU=0
BATCH_SIZES="1,2,4,8"
NUM_REQUESTS="16"
ARRIVAL_RATES="0.0"
OUTPUT_LOW=8
OUTPUT_HIGH=64
MAX_PROMPT_TOKENS=512
TTFT_SLO=2.0
TPOT_SLO=0.05
INTERFERENCE_FACTOR=1.3
BASELINE_ONLY=""
SKIP_PLOTS=false
SHAREGPT_JSONL=""
OUT_DIR="results/milestone/workload_sweep"
SEED=42

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)             MODEL="$2";             shift 2 ;;
    --prefill-gpu)       PREFILL_GPU="$2";       shift 2 ;;
    --decode-gpu)        DECODE_GPU="$2";        shift 2 ;;
    --baseline-gpu)      BASELINE_GPU="$2";      shift 2 ;;
    --batch-sizes)       BATCH_SIZES="$2";       shift 2 ;;
    --num-requests)      NUM_REQUESTS="$2";      shift 2 ;;
    --arrival-rates)     ARRIVAL_RATES="$2";     shift 2 ;;
    --output-low)        OUTPUT_LOW="$2";        shift 2 ;;
    --output-high)       OUTPUT_HIGH="$2";       shift 2 ;;
    --max-prompt-tokens) MAX_PROMPT_TOKENS="$2"; shift 2 ;;
    --ttft-slo)            TTFT_SLO="$2";             shift 2 ;;
    --tpot-slo)            TPOT_SLO="$2";             shift 2 ;;
    --interference-factor) INTERFERENCE_FACTOR="$2";  shift 2 ;;
    --baseline-only)       BASELINE_ONLY="--baseline-only"; shift ;;
    --skip-plots)        SKIP_PLOTS=true;        shift ;;
    --sharegpt-jsonl)    SHAREGPT_JSONL="$2";    shift 2 ;;
    --out-dir)           OUT_DIR="$2";           shift 2 ;;
    --seed)              SEED="$2";              shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

JSONL_FLAG=""
if [[ -n "$SHAREGPT_JSONL" ]]; then
  JSONL_FLAG="--sharegpt-jsonl $SHAREGPT_JSONL"
fi

echo "======================================================================"
echo "  GPU Workload Sweep (batch × num_requests × arrival_rate)"
echo "  Model:          $MODEL"
echo "  Batch sizes:    $BATCH_SIZES"
echo "  num_requests:   $NUM_REQUESTS"
echo "  arrival_rates:  $ARRIVAL_RATES"
echo "  Output tokens:  [$OUTPUT_LOW, $OUTPUT_HIGH]"
echo "  Prefill GPU:    $PREFILL_GPU"
echo "  Decode GPU:     $DECODE_GPU"
echo "  Baseline only:  ${BASELINE_ONLY:-no}"
echo "  Out dir:        $OUT_DIR"
echo "======================================================================"
echo ""

echo "Python:  $(command -v python)"
echo "Version: $(python --version 2>&1)"
echo ""

# Probe torch + CUDA without aborting on non-zero exit.
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
  echo ""
  echo "Only 1 GPU visible; forcing --baseline-only."
  BASELINE_ONLY="--baseline-only"
fi

echo ""
echo ">>> Running workload sweep..."
python -m src.experiments.run_workload_sweep_gpu \
  --model "$MODEL" \
  --batch-sizes "$BATCH_SIZES" \
  --num-requests "$NUM_REQUESTS" \
  --arrival-rates "$ARRIVAL_RATES" \
  --baseline-gpu "$BASELINE_GPU" \
  --prefill-gpu "$PREFILL_GPU" \
  --decode-gpu "$DECODE_GPU" \
  --output-low "$OUTPUT_LOW" \
  --output-high "$OUTPUT_HIGH" \
  --max-prompt-tokens "$MAX_PROMPT_TOKENS" \
  --ttft-slo "$TTFT_SLO" \
  --tpot-slo "$TPOT_SLO" \
  --interference-factor "$INTERFERENCE_FACTOR" \
  --seed "$SEED" \
  --out-dir "$OUT_DIR" \
  $BASELINE_ONLY \
  $JSONL_FLAG

SUMMARY_CSV="$OUT_DIR/summary.csv"
if [[ ! -f "$SUMMARY_CSV" ]]; then
  echo "ERROR: expected $SUMMARY_CSV but it was not produced." >&2
  exit 1
fi

if [[ "$SKIP_PLOTS" == "true" ]]; then
  echo ""
  echo "Plotting skipped (--skip-plots). Summary at $SUMMARY_CSV."
  exit 0
fi

echo ""
echo ">>> Generating plots from $SUMMARY_CSV ..."
python -m src.experiments.plot_workload_sweep_gpu \
  --csv "$SUMMARY_CSV" \
  --out-dir "$OUT_DIR/plots"

echo ""
echo "======================================================================"
echo "  Done."
echo "  Summary CSV:      $SUMMARY_CSV"
echo "  Per-request CSVs: $OUT_DIR/requests/"
echo "  Conditions JSONL: $OUT_DIR/conditions.jsonl"
echo "  Meta:             $OUT_DIR/meta.json"
echo "  Log:              $OUT_DIR/run.log"
echo "  Plots:            $OUT_DIR/plots/"
echo "======================================================================"
