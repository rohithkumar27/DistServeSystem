#!/usr/bin/env bash
# =============================================================================
# run_milestone.sh — Run all milestone experiments and collect results.
#
# Simulator experiments run on CPU (no GPU required).
# GPU experiments run automatically if CUDA is available; skip with --sim-only.
#
# Usage:
#   bash run_milestone.sh                        # simulator + GPU (if available)
#   bash run_milestone.sh --sim-only             # simulator experiments only
#   bash run_milestone.sh --timing-json PATH     # use fitted timing from Stage B
#   bash run_milestone.sh --model MODEL          # HF model for GPU experiments
#   bash run_milestone.sh --prefill-gpu 0 --decode-gpu 1
#   bash run_milestone.sh --num-gpu-requests 32  # requests per GPU run (keep small)
# =============================================================================

set -euo pipefail

# ---------- defaults ----------------------------------------------------------
TIMING_JSON=""
SIM_ONLY=false
MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PREFILL_GPU=0
DECODE_GPU=1
NUM_GPU_REQUESTS=32
BATCH_SIZES="1,2,4,8"
SHAREGPT_JSONL=""

# ---------- parse args --------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --sim-only)          SIM_ONLY=true;         shift ;;
    --timing-json)       TIMING_JSON="$2";      shift 2 ;;
    --model)             MODEL="$2";            shift 2 ;;
    --prefill-gpu)       PREFILL_GPU="$2";      shift 2 ;;
    --decode-gpu)        DECODE_GPU="$2";       shift 2 ;;
    --num-gpu-requests)  NUM_GPU_REQUESTS="$2"; shift 2 ;;
    --batch-sizes)       BATCH_SIZES="$2";      shift 2 ;;
    --sharegpt-jsonl)    SHAREGPT_JSONL="$2";   shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# ---------- timing-json flag for Python scripts --------------------------------
TIMING_FLAG=""
if [[ -n "$TIMING_JSON" ]]; then
  TIMING_FLAG="--timing-json $TIMING_JSON"
fi

JSONL_FLAG=""
if [[ -n "$SHAREGPT_JSONL" ]]; then
  JSONL_FLAG="--sharegpt-jsonl $SHAREGPT_JSONL"
fi

RESULTS_DIR="results/milestone"
mkdir -p "$RESULTS_DIR"

echo "======================================================================"
echo "  DistServeSystem Milestone Experiments"
echo "  Output directory: $RESULTS_DIR"
echo "  Timing source:    ${TIMING_JSON:-synthetic defaults}"
echo "  Sim-only mode:    $SIM_ONLY"
echo "======================================================================"
echo ""

# ---------- 1. Core comparison (baseline vs disaggregated) --------------------
echo ">>> [1/5] Core simulator comparison (400 requests, bursty mixed workload)"
python -m src.experiments.run_comparison \
  --num-requests 400 --arrival-rate 2.0 \
  --ttft-slo 0.8 --tpot-slo 0.03 \
  $TIMING_FLAG
echo ""

# ---------- 2. Load sweep (arrival rate vs goodput curve) ---------------------
echo ">>> [2/5] Load sweep: arrival rate 0.5 → 4.0 req/s"
python -m src.experiments.run_load_sweep \
  --num-requests 500 \
  --ttft-slo 0.8 --tpot-slo 0.03 \
  $TIMING_FLAG
echo ""

# ---------- 3. Workload mix sweep (interactive fraction sweep) ----------------
echo ">>> [3/5] Workload mix sweep: interactive fraction 0.0 → 1.0"
python -m src.experiments.run_workload_mix \
  --num-requests 500 --arrival-rate 2.0 \
  --ttft-slo 0.8 --tpot-slo 0.03 \
  $TIMING_FLAG
echo ""

# ---------- 4. Ablations (KV transfer + capacity split) ----------------------
echo ">>> [4/5] Ablations: KV transfer cost sweep + capacity split sweep"
python -m src.experiments.run_ablations $TIMING_FLAG
echo ""

# ---------- 5. GPU experiments (optional) ------------------------------------
if [[ "$SIM_ONLY" == "true" ]]; then
  echo ">>> [5/5] GPU experiments: SKIPPED (--sim-only)"
else
  # Check for CUDA via Python
  if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
    echo ">>> [5/5] GPU experiments: $NUM_GPUS GPU(s) detected"

    if [[ "$NUM_GPUS" -ge 2 ]]; then
      echo "    Running full colocated + disaggregated GPU comparison..."
      python -m src.experiments.run_gpu_comparison \
        --model "$MODEL" \
        --num-requests "$NUM_GPU_REQUESTS" \
        --prefill-gpu "$PREFILL_GPU" \
        --decode-gpu "$DECODE_GPU" \
        --ttft-slo 2.0 --tpot-slo 0.05 \
        $JSONL_FLAG

      echo "    Running GPU batch-size sweep (colocated + disaggregated)..."
      python -m src.experiments.run_batch_sweep_gpu \
        --model "$MODEL" \
        --num-requests "$NUM_GPU_REQUESTS" \
        --prefill-gpu "$PREFILL_GPU" \
        --decode-gpu "$DECODE_GPU" \
        --batch-sizes "$BATCH_SIZES" \
        --ttft-slo 2.0 --tpot-slo 0.05 \
        $JSONL_FLAG
    else
      echo "    Only 1 GPU available — running baseline-only GPU experiments..."
      python -m src.experiments.run_gpu_comparison \
        --model "$MODEL" \
        --num-requests "$NUM_GPU_REQUESTS" \
        --baseline-gpu "$PREFILL_GPU" \
        --prefill-gpu "$PREFILL_GPU" \
        --decode-gpu "$PREFILL_GPU" \
        --ttft-slo 2.0 --tpot-slo 0.05 \
        --baseline-only \
        $JSONL_FLAG

      python -m src.experiments.run_batch_sweep_gpu \
        --model "$MODEL" \
        --num-requests "$NUM_GPU_REQUESTS" \
        --baseline-gpu "$PREFILL_GPU" \
        --prefill-gpu "$PREFILL_GPU" \
        --decode-gpu "$PREFILL_GPU" \
        --batch-sizes "$BATCH_SIZES" \
        --ttft-slo 2.0 --tpot-slo 0.05 \
        --baseline-only \
        $JSONL_FLAG
    fi
  else
    echo ">>> [5/5] GPU experiments: SKIPPED (no CUDA device found)"
    echo "    Install PyTorch with CUDA or run on a GPU node."
  fi
fi

# ---------- manifest ----------------------------------------------------------
echo ""
echo "======================================================================"
echo "  Results written to: $RESULTS_DIR  and  results/"
echo "======================================================================"
echo ""
echo "Simulator outputs (results/milestone/):"
for f in "$RESULTS_DIR"/*.csv "$RESULTS_DIR"/*.json; do
  [[ -f "$f" ]] && echo "  $f"
done

echo ""
echo "Timestamped run outputs (results/):"
for f in results/*.csv results/*.json; do
  [[ -f "$f" ]] && echo "  $f"
done

echo ""
echo "Suggested columns for report tables:"
echo "  load_sweep.csv     → arrival_rate, coloc_goodput, disagg_goodput, coloc_p95_ttft, disagg_p95_ttft"
echo "  workload_mix.csv   → interactive_frac, coloc_goodput, disagg_goodput, goodput_gain_pct"
echo "  ablations CSV      → ablation, transfer_per_prompt_token / capacity mults, goodput columns"
echo "  gpu_batch_sweep.csv → design, batch_size, goodput, mean_ttft, throughput_req_s"
echo ""
echo "Done."
