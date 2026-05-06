import os
import re
import json
import argparse
from triton_validator import check_triton_validity

try:
    import torch
except Exception:
    torch = None

from golden_path import resolve_golden_results_dir

_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
_PERF_DIR = os.path.normpath(os.path.join(_EVAL_DIR, "../metrics"))
_REF_DIR = resolve_golden_results_dir()
_GOLDEN_METRICS_DIR = os.path.join(_PERF_DIR, "golden_metrics")
_DEFAULT_DTYPE = "float32"


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _infer_dtype_from_name(op_name: str):
    name = op_name.lower()
    for token in ("fp16", "bf16", "fp32", "fp64", "int8", "int4", "w8a8", "w4a16"):
        if token in name:
            return token
    return ""


def _extract_dtype(file_json: str):
    op_name = file_json.replace(".json", "")
    perf_py = os.path.join(_GOLDEN_METRICS_DIR, f"{op_name}_perf.py")
    name_guess = _infer_dtype_from_name(op_name)
    if not os.path.exists(perf_py):
        return name_guess or _DEFAULT_DTYPE
    text = open(perf_py, "r", encoding="utf-8").read()
    m = re.search(r"def __init__\s*\(\s*self\s*,\s*dtype\s*=\s*([^,\)\n]+)", text)
    if not m:
        return name_guess or _DEFAULT_DTYPE
    raw = m.group(1).strip()
    if raw.startswith("torch."):
        return raw.split(".", 1)[1]
    if raw in ("None", "NoneType", "null"):
        return name_guess or _DEFAULT_DTYPE
    return raw or name_guess or _DEFAULT_DTYPE


def _aggregate_perf(data_gen, file_name):
    if not isinstance(data_gen, list) or len(data_gen) == 0:
        raise ValueError(
            f"{file_name}: empty generated perf JSON (no valid timing rows in perf_results; "
            f"often means all input sizes failed in run_benchmark). This is independent of the golden reference JSON."
        )

    total_runtime_ms = 0.0
    total_bytes_gb = 0.0
    total_flops_t = 0.0

    for idx, item in enumerate(data_gen):
        ms = _to_float(item.get("ms"))
        gbs = _to_float(item.get("GB/s"))
        tflops = _to_float(item.get("TFLOPS"))
        if ms is None or gbs is None or tflops is None:
            raise ValueError(f"{file_name}: invalid perf row index={idx}, require ms/GB/s/TFLOPS")
        if ms <= 0:
            raise ValueError(f"{file_name}: invalid ms={ms} at row index={idx}")

        total_runtime_ms += ms
        total_bytes_gb += gbs * (ms / 1000.0)
        total_flops_t += tflops * (ms / 1000.0)

    agg_gbs = total_bytes_gb / total_runtime_ms * 1000.0
    agg_tflops = total_flops_t / (total_runtime_ms / 1000.0)
    return {
        "runtime_ms": round(total_runtime_ms, 4),
        "gbs": round(agg_gbs, 4),
        "tflops": round(agg_tflops, 4),
    }


def _calculate_speedup_and_metrics(path_gen, path_ref, file_name):
    data_gen = json.load(open(path_gen, "r", encoding="utf-8"))
    data_ref = json.load(open(path_ref, "r", encoding="utf-8"))
    metrics = _aggregate_perf(data_gen, file_name)
    if not isinstance(data_ref, list) or len(data_ref) == 0:
        raise ValueError(f"{file_name}: empty reference data")

    total_ms_ref = 0.0
    for idx, item in enumerate(data_ref):
        ms = _to_float(item.get("ms"))
        if ms is None:
            raise ValueError(f"{file_name}: invalid reference row index={idx}, missing ms")
        if ms <= 0:
            raise ValueError(f"{file_name}: invalid reference ms={ms} at row index={idx}")
        total_ms_ref += ms

    speedup = total_ms_ref / metrics["runtime_ms"]
    return round(speedup, 4), metrics, total_ms_ref, metrics["runtime_ms"]


def statis(gen_folder, result_jsonl=None):
    files = [f for f in os.listdir(gen_folder) if f.endswith(".json")]
    results = []
    speedups = []
    total_ref_ms_sum = 0.0
    total_gen_ms_sum = 0.0
    
    # Thresholds for filtering anomalous values.
    ABNORMAL_RUNTIME_THRESHOLD_MS = 10000  # >10s total runtime is considered abnormally slow
    ABNORMAL_SPEEDUP_THRESHOLD = 0.01      # speedup < 0.01x is considered abnormally slow

    print("\n" + "=" * 60)
    print("Performance Results:")
    print("=" * 60)

    for f in files:
        path_gen = os.path.join(gen_folder, f)
        path_ref = os.path.join(_REF_DIR, f)
        file_py = f.replace(".json", ".py")
        code_path = path_gen.replace(".json", ".py")
        code = open(code_path, "r", encoding="utf-8").read() if os.path.exists(code_path) else ""
        dtype = _extract_dtype(f)

        fail_row = {
            "file": file_py,
            "ok": False,
            "dtype": dtype,
            "runtime_ms": "N/A",
            "gbs": "N/A",
            "tflops": "N/A",
            "speedup": None,
            "error": "",
            "code": code,
        }

        if not os.path.exists(path_ref):
            print(f"  {file_py}: SKIP (no reference)")
            fail_row["error"] = "no reference"
            results.append(fail_row)
            continue

        try:
            speedup, metrics, total_ms_ref, total_ms_gen = _calculate_speedup_and_metrics(path_gen, path_ref, file_py)
        except Exception as e:
            print(f"  {file_py}: FAIL ({e})")
            fail_row["error"] = str(e)
            results.append(fail_row)
            continue

        # Detect abnormal performance (very long runtime or extremely low speedup).
        is_abnormal = (metrics["runtime_ms"] > ABNORMAL_RUNTIME_THRESHOLD_MS or 
                      speedup < ABNORMAL_SPEEDUP_THRESHOLD)
        
        # Code validity check
        validity = {}
        if check_triton_validity and code:
            validity = check_triton_validity(code, speedup)
            if validity.get("is_suspect") or is_abnormal:
                flags = validity.get("flags", [])
                if is_abnormal:
                    flags.append(f"abnormal_perf(runtime={metrics['runtime_ms']:.1f}ms,speedup={speedup:.4f}x)")
                print(f"  {file_py}: {speedup:.4f}x ⚠️ {flags}")
            else:
                print(f"  {file_py}: {speedup:.4f}x")
        else:
            if is_abnormal:
                print(f"  {file_py}: {speedup:.4f}x ⚠️ ABNORMAL (runtime={metrics['runtime_ms']:.1f}ms)")
            else:
                print(f"  {file_py}: {speedup:.4f}x")
        
        # Only include non-anomalous tasks in the average speedup statistics.
        if not is_abnormal:
            speedups.append(speedup)
            total_ref_ms_sum += total_ms_ref
            total_gen_ms_sum += total_ms_gen
        results.append(
            {
                "file": file_py,
                "ok": True,
                "dtype": dtype,
                "runtime_ms": metrics["runtime_ms"],
                "gbs": metrics["gbs"],
                "tflops": metrics["tflops"],
                "speedup": speedup,
                "error": "",
                "code": code,
                "validity_flags": validity.get("flags", []),
                "is_suspect": validity.get("is_suspect", False),
            }
        )

    successful_count = sum(1 for r in results if r.get("ok", False))
    abnormal_count = successful_count - len(speedups)
    
    if speedups:
        # Use the mean speedup over non-anomalous tasks (anomalies excluded).
        avg_speedup = sum(speedups) / len(speedups)
        # Also compute a global time-weighted speedup for reference.
        total_time_speedup = total_ref_ms_sum / total_gen_ms_sum if total_gen_ms_sum > 0 else 0
        print("\n" + "-" * 60)
        print(f"Average Speedup (normal cases): {avg_speedup:.4f}x ({len(speedups)}/{len(files)} passed)")
        if abnormal_count > 0:
            print(f"Abnormal cases excluded: {abnormal_count} (runtime>{ABNORMAL_RUNTIME_THRESHOLD_MS}ms or speedup<{ABNORMAL_SPEEDUP_THRESHOLD}x)")
        print(f"Total Time Speedup (all): {total_time_speedup:.4f}x")
    else:
        print("\nNo successful benchmarks")
    print("=" * 60 + "\n")

    if result_jsonl:
        os.makedirs(os.path.dirname(result_jsonl) or ".", exist_ok=True)
        with open(result_jsonl, "w", encoding="utf-8") as f_out:
            for r in results:
                f_out.write(json.dumps(r, ensure_ascii=False) + "\n")

    return results


def arg_parser():
    parser = argparse.ArgumentParser(description="Efficiency statistics")
    parser.add_argument("--gen_folder", type=str, required=True, help="The generated folder path")
    parser.add_argument("--result_jsonl", type=str, default=None, help="Path to output structured results.")
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    statis(args.gen_folder, args.result_jsonl)