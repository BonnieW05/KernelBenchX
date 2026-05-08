#!/bin/bash
# Evaluate generated .py files directly without needing JSONL conversion
# Usage: ./run_eval_from_files.sh [--bench-timeout SEC] <input_dir_or_file.py> <output_dir> <gpus>
#
# Examples:
#   ./run_eval_from_files.sh /path/to/generated/kernel.py ./results 0
#   ./run_eval_from_files.sh /path/to/generated/kernels/ ./results 0,1,2
#   ./run_eval_from_files.sh --bench-timeout 300 ./kernels ./results 0

set -e

BENCH_TIMEOUT="${KERNELBENCHX_BENCH_TIMEOUT:-180}"
PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python}}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    if command -v python >/dev/null 2>&1; then
        PYTHON_BIN="python"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="python3"
    else
        echo "No python interpreter found (tried: \$PYTHON_BIN / python / python3)" >&2
        exit 127
    fi
fi

while [[ "${1:-}" == --* ]]; do
    case "$1" in
        --bench-timeout)
            BENCH_TIMEOUT="${2:?Missing value for --bench-timeout}"
            shift 2
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

INPUT_PATH="${1:?Usage: $0 [options] <input_dir_or_file.py> <output_dir> <gpus>}"
OUTPUT_DIR="${2:?Usage: $0 [options] <input_dir_or_file.py> <output_dir> <gpus>}"
GPUS="${3:-0}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EVAL_DIR="$SCRIPT_DIR/../EVAL"
PERF_DIR="$SCRIPT_DIR/../metrics"

# Convert to absolute path.
case "$OUTPUT_DIR" in
    /*) ;;
    *) OUTPUT_DIR="$(pwd)/$OUTPUT_DIR" ;;
esac

mkdir -p "$OUTPUT_DIR"
INTERMEDIATE_DIR="$OUTPUT_DIR/intermediate"
mkdir -p "$INTERMEDIATE_DIR"

CALL_JSONL="$INTERMEDIATE_DIR/results_call.jsonl"
EXE_JSONL="$INTERMEDIATE_DIR/results_exe.jsonl"
EFF_JSONL="$INTERMEDIATE_DIR/results_eff.jsonl"

echo "=== Stage 1: Call Accuracy ==="
"$PYTHON_BIN" "$EVAL_DIR/0_call_acc.py" \
    --source "$INPUT_PATH" \
    --target "$INTERMEDIATE_DIR/call_acc" \
    --GPUs "$GPUS" \
    --result_jsonl "$CALL_JSONL"

echo "=== Stage 2: Execution Accuracy ==="
"$PYTHON_BIN" "$EVAL_DIR/1_exe_acc.py" \
    --folder "$INTERMEDIATE_DIR/call_acc" \
    --GPUs "$GPUS" \
    --result_jsonl "$EXE_JSONL"

# Merge per-subfolder results if they exist
if ls "$INTERMEDIATE_DIR"/results_exe_*.jsonl >/dev/null 2>&1; then
    : > "$EXE_JSONL"
    for _f in "$INTERMEDIATE_DIR"/results_exe_*.jsonl; do
        [ -f "$_f" ] && cat "$_f" >> "$EXE_JSONL"
    done
    # Remove per-subfolder exe shards after merge.
    rm -f "$INTERMEDIATE_DIR"/results_exe_*.jsonl
fi

_EXE_PASSED=$(python - "$EXE_JSONL" <<'PY'
import json
import pathlib
import sys

src = pathlib.Path(sys.argv[1])
passed = 0
if src.exists():
    for line in src.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if json.loads(line).get("ok"):
            passed += 1
print(passed)
PY
)

echo "=== Stage 3: Efficiency (${_EXE_PASSED} kernel(s) passed call+exe) ==="
PERF_INPUT_DIR="$INTERMEDIATE_DIR/call_acc"
PERF_RESULTS_DIR="$INTERMEDIATE_DIR/perf_results"
PERF_SCRIPT_DIR="$INTERMEDIATE_DIR/perf_scripts"
PERF_LOG_DIR="$INTERMEDIATE_DIR/perf_logs"

mkdir -p "$PERF_RESULTS_DIR" "$PERF_SCRIPT_DIR" "$PERF_LOG_DIR"

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
        "$PYTHON_BIN" run_bench/write_file.py \
            --input_folder_path "$PERF_INPUT_DIR" \
            --results_path "$PERF_RESULTS_DIR" \
            --exe_jsonl "$EXE_JSONL")

    (cd "$PERF_DIR" && \
        CUDA_VISIBLE_DEVICES="$CVD_GPUS" \
        KERNELBENCHX_SCRIPT_DIR="$PERF_SCRIPT_DIR" \
        KERNELBENCHX_LOG_DIR="$PERF_LOG_DIR" \
        KERNELBENCHX_RESULTS_PATH="$PERF_RESULTS_DIR" \
        KERNELBENCHX_BENCH_TIMEOUT="$BENCH_TIMEOUT" \
        "$PYTHON_BIN" run_bench/multiprocess_gpu_run.py)

    "$PYTHON_BIN" "$EVAL_DIR/2_efficiency.py" \
        --gen_folder "$PERF_RESULTS_DIR" \
        --result_jsonl "$EFF_JSONL"
fi

echo "=== Creating Unified Results ==="
UNIFIED_JSON="$OUTPUT_DIR/metrics.json"
"$PYTHON_BIN" "$EVAL_DIR/unified_results.py" \
    --call "$CALL_JSONL" \
    --exe "$EXE_JSONL" \
    --eff "$EFF_JSONL" \
    --perf_dir "$PERF_RESULTS_DIR" \
    --output "$UNIFIED_JSON"

echo ""
echo "=== Evaluation Complete ==="
echo "Results: $OUTPUT_DIR"
echo "  - metrics.json (complete data)"
echo "  - summary.json (statistics)"
echo "  - intermediate/ (temp files)"
