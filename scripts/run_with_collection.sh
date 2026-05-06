#!/bin/bash
# Evaluation with data collection for iterative experiments
# Usage: ./run_with_collection.sh [--bench-timeout SEC] [--model NAME] [--iter ID] <input.jsonl> <output_dir> <gpus> [model] [iter]

set -e

BENCH_TIMEOUT="${KERNELBENCHX_BENCH_TIMEOUT:-180}"
CONTROLLER_DIR="${KERNELBENCHX_CONTROLLER_DIR:-}"
KEEP_INTERMEDIATE="${KERNELBENCHX_KEEP_INTERMEDIATE:-1}"

MODEL=""
ITER="0"
_MODEL_SET=0
_ITER_SET=0

while [[ "${1:-}" == --* ]]; do
    case "$1" in
        --bench-timeout)
            BENCH_TIMEOUT="${2:?Missing value for --bench-timeout}"
            shift 2
            ;;
        --model)
            MODEL="${2:?Missing value for --model}"
            _MODEL_SET=1
            shift 2
            ;;
        --iter)
            ITER="${2:?Missing value for --iter}"
            _ITER_SET=1
            shift 2
            ;;
        --controller-dir)
            CONTROLLER_DIR="${2:?Missing value for --controller-dir}"
            shift 2
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

INPUT_JSONL="${1:?Usage: $0 [options] <input.jsonl> <output_dir> <gpus> [model] [iter]}"
OUTPUT_DIR="${2:?Usage: $0 [options] <input.jsonl> <output_dir> <gpus> [model] [iter]}"
GPUS="${3:-0}"

if [[ "$_MODEL_SET" -eq 0 ]]; then
    MODEL="${4:-}"
fi
if [[ "$_ITER_SET" -eq 0 ]]; then
    ITER="${5:-0}"
fi
if [[ -z "$CONTROLLER_DIR" ]]; then
    CONTROLLER_DIR="$OUTPUT_DIR"
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EVAL_DIR="$SCRIPT_DIR/../EVAL"
PERF_DIR="$SCRIPT_DIR/../metrics"

mkdir -p "$OUTPUT_DIR"

CALL_JSONL="$OUTPUT_DIR/results_call.jsonl"
EXE_JSONL="$OUTPUT_DIR/results_exe.jsonl"
EFF_JSONL="$OUTPUT_DIR/results_eff.jsonl"
FAILED_EXE_JSONL="$OUTPUT_DIR/results_exe_failed.jsonl"

echo "=== Stage 1: Call Accuracy ==="
python "$EVAL_DIR/0_call_acc.py" \
    --source "$INPUT_JSONL" \
    --target "$OUTPUT_DIR/call_acc" \
    --GPUs "$GPUS" \
    --result_jsonl "$CALL_JSONL"

echo "=== Stage 2: Execution Accuracy ==="
python "$EVAL_DIR/1_exe_acc.py" \
    --folder "$OUTPUT_DIR/call_acc" \
    --GPUs "$GPUS" \
    --result_jsonl "$EXE_JSONL"

# 1_exe_acc.py writes per-subfolder files like results_exe_Math.jsonl.
# Merge them all into the single EXE_JSONL expected by later stages.
for _f in "$OUTPUT_DIR"/results_exe_*.jsonl; do
    [ -f "$_f" ] && cat "$_f" >> "$EXE_JSONL"
done

# Export failed execution cases for debugging and count passed exe cases.
_EXE_PASSED=$(python - "$EXE_JSONL" "$FAILED_EXE_JSONL" <<'PY'
import json
import pathlib
import sys
src = pathlib.Path(sys.argv[1])
dst = pathlib.Path(sys.argv[2])
dst.parent.mkdir(parents=True, exist_ok=True)
if not src.exists():
    dst.write_text("", encoding="utf-8")
    print(0)
    raise SystemExit(0)
passed = 0
failed = []
for line in src.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line:
        continue
    item = json.loads(line)
    if item.get("ok"):
        passed += 1
    else:
        failed.append(item)
with dst.open("w", encoding="utf-8") as f:
    for item in failed:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
print(passed)
PY
)

# NOTE: We intentionally do not copy historical failure artifacts from a global
# kernelbenchx/test_failures/ directory into this run's OUTPUT_DIR. For per-task
# failure tensor details, use this run's 1_exe_acc outputs / unified metrics. A
# global directory (if any) is treated as a developer-local scratch space and is
# not part of the experiment artifact.

echo "=== Stage 3: Efficiency (${_EXE_PASSED} kernel(s) passed call+exe) ==="
PERF_INPUT_DIR="$OUTPUT_DIR/call_acc"
PERF_RESULTS_DIR="$OUTPUT_DIR/perf_results"
PERF_SCRIPT_DIR="$OUTPUT_DIR/perf_scripts"
PERF_LOG_DIR="$OUTPUT_DIR/perf_logs"

if [ "$_EXE_PASSED" -eq 0 ]; then
    echo "Skipping Stage 3: no kernels passed execution accuracy."
    touch "$EFF_JSONL"
else
    CVD_GPUS="$GPUS"
    CVD_GPUS="${CVD_GPUS#[}"
    CVD_GPUS="${CVD_GPUS%]}"
    CVD_GPUS="${CVD_GPUS// /}"

    (cd "$PERF_DIR" && \
        KERNELBENCHX_SCRIPT_DIR="$PERF_SCRIPT_DIR" \
        KERNELBENCHX_LOG_DIR="$PERF_LOG_DIR" \
        python run_bench/write_file.py \
            --input_folder_path "$PERF_INPUT_DIR" \
            --results_path "$PERF_RESULTS_DIR")

    (cd "$PERF_DIR" && \
        CUDA_VISIBLE_DEVICES="$CVD_GPUS" \
        KERNELBENCHX_SCRIPT_DIR="$PERF_SCRIPT_DIR" \
        KERNELBENCHX_LOG_DIR="$PERF_LOG_DIR" \
        KERNELBENCHX_RESULTS_PATH="$PERF_RESULTS_DIR" \
        KERNELBENCHX_BENCH_TIMEOUT="$BENCH_TIMEOUT" \
        python run_bench/multiprocess_gpu_run.py)

    python "$EVAL_DIR/2_efficiency.py" \
        --gen_folder "$PERF_RESULTS_DIR" \
        --result_jsonl "$EFF_JSONL"
fi

echo "=== Processing with Controller ==="
export KERNELBENCHX_COLLECT_DIR="$CONTROLLER_DIR/collected_data"
python "$EVAL_DIR/controller.py" \
    --iter "$ITER" \
    --call "$CALL_JSONL" \
    --exe "$EXE_JSONL" \
    --eff "$EFF_JSONL" \
    --model "$MODEL" \
    --collect 1 \
    --output_dir "$CONTROLLER_DIR"

echo "=== Creating Unified Results ==="
UNIFIED_JSON="$OUTPUT_DIR/metrics.json"
python "$EVAL_DIR/unified_results.py" \
    --call "$CALL_JSONL" \
    --exe "$EXE_JSONL" \
    --eff "$EFF_JSONL" \
    --perf_dir "$PERF_RESULTS_DIR" \
    --output "$UNIFIED_JSON"

if [ "$KEEP_INTERMEDIATE" = "0" ]; then
    rm -f "$CALL_JSONL" "$EFF_JSONL"
    rm -f "$OUTPUT_DIR"/results_exe_*.jsonl
    rm -rf "$PERF_RESULTS_DIR" "$PERF_SCRIPT_DIR" "$PERF_LOG_DIR"
fi

echo "=== Complete (iter=$ITER, model=$MODEL) ==="
echo "Scripts: $OUTPUT_DIR/call_acc/"
echo "Metrics: $UNIFIED_JSON"
if [ -d "$OUTPUT_DIR/collected_data" ]; then
    echo "Collected: $OUTPUT_DIR/collected_data/"
fi
