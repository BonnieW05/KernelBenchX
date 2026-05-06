#!/usr/bin/env python3
import json
import os
import argparse

from golden_path import resolve_golden_results_dir

_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
_METADATA_PATH = os.path.join(_EVAL_DIR, "../data/kernelbenchx_v1.json")


def _build_basename_to_fullpath():
    """Build a mapping from bare filename (e.g. 'sqrt.py') to full relative path
    (e.g. 'Math/sqrt.py') using the benchmark metadata."""
    mapping = {}
    if not os.path.exists(_METADATA_PATH):
        return mapping
    try:
        with open(_METADATA_PATH, "r", encoding="utf-8") as f:
            items = json.load(f)
        for item in items:
            full = item.get("file", "")
            if full:
                mapping[os.path.basename(full)] = full
    except Exception:
        pass
    return mapping


def _read_perf_log_error(perf_results_dir, op_name):
    """Read error message from perf_logs/*.err files if available."""
    if not perf_results_dir:
        return ""
    # perf_logs is sibling to perf_results
    logs_dir = os.path.join(os.path.dirname(perf_results_dir), "perf_logs")
    if not os.path.isdir(logs_dir):
        return ""
    # Try both <op>_perf.py.err and <op>.py.err
    candidates = [
        os.path.join(logs_dir, f"{op_name}_perf.py.err"),
        os.path.join(logs_dir, f"{op_name}.py.err"),
    ]
    for err_path in candidates:
        if os.path.exists(err_path):
            try:
                content = open(err_path, 'r', encoding='utf-8', errors='replace').read().strip()
                if content:
                    # Truncate to avoid huge error messages
                    return content[:500]
            except Exception:
                pass
    return ""


def merge_results(call_jsonl, exe_jsonl, eff_jsonl, perf_results_dir, output_json):
    """Merge Call/Exe/Perf results into a unified JSON with table-friendly fields."""

    # basename -> fullpath, e.g. 'sqrt.py' -> 'Math/sqrt.py'
    base_to_full = _build_basename_to_fullpath()

    def _normalize_key(file_key):
        """Normalize a bare filename to its full relative path if possible."""
        if os.sep not in file_key and "/" not in file_key:
            return base_to_full.get(file_key, file_key)
        return file_key

    def load_jsonl(path, normalize=False):
        if not os.path.exists(path):
            return {}
        result = {}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                key = _normalize_key(item["file"]) if normalize else item["file"]
                result[key] = item
        return result

    def load_perf_data(perf_dir):
        perf_data = {}
        if not os.path.exists(perf_dir):
            return perf_data
        for fname in os.listdir(perf_dir):
            if fname.endswith('.json'):
                fpath = os.path.join(perf_dir, fname)
                with open(fpath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                py_name = fname.replace('.json', '.py')
                # Normalize to full path
                key = _normalize_key(py_name)
                perf_data[key] = data
        return perf_data

    def _sum_ms_from_perf_list(perf_list):
        """Sum of benchmark row times (ms) from a perf JSON list."""
        s = 0.0
        if not isinstance(perf_list, list):
            return s
        for item in perf_list:
            ms = item.get("ms")
            if isinstance(ms, (int, float)) and ms > 0:
                s += float(ms)
        return s

    def load_golden_ref_ms_map():
        """basename.json -> total reference ms from golden_results (for avg_speedup only)."""
        ref_map = {}
        golden_dir = resolve_golden_results_dir()
        if not os.path.isdir(golden_dir):
            return ref_map
        for fname in os.listdir(golden_dir):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(golden_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                py_name = fname.replace(".json", ".py")
                key = _normalize_key(py_name)
                ref_map[key] = _sum_ms_from_perf_list(data)
            except Exception:
                continue
        return ref_map

    def _is_number(v):
        return isinstance(v, (int, float))

    def _na_if_needed(value, use_na):
        if use_na:
            return "N/A"
        return value

    # call_acc writes full paths; exe/eff write bare filenames — normalize the latter
    call_results = load_jsonl(call_jsonl, normalize=False) if call_jsonl else {}
    exe_results = load_jsonl(exe_jsonl, normalize=True) if exe_jsonl else {}
    eff_results = load_jsonl(eff_jsonl, normalize=True) if eff_jsonl else {}
    perf_data = load_perf_data(perf_results_dir) if perf_results_dir else {}
    golden_ref_ms = load_golden_ref_ms_map()

    all_files = set(call_results.keys()) | set(exe_results.keys()) | set(eff_results.keys())

    unified = {}
    for fname in all_files:
        call_data = call_results.get(fname, {})
        exe_data = exe_results.get(fname, {})
        eff_data = eff_results.get(fname, {})
        perf = perf_data.get(fname, [])

        compile_ok = bool(call_data.get("ok", False))
        correctness_ok = bool(exe_data.get("ok", False))
        # Perf is meaningful only when execution is correct.
        perf_ok = bool(eff_data.get("ok", False)) and correctness_ok
        should_na = (not perf_ok) or (not correctness_ok)

        code_quality = call_data.get("code_quality", {})
        maintainability = code_quality.get("maintainability_index", "N/A")
        if not _is_number(maintainability):
            maintainability = "N/A"

        runtime_ms = eff_data.get("runtime_ms", "N/A")
        gbs = eff_data.get("gbs", "N/A")
        tflops = eff_data.get("tflops", "N/A")
        speedup = eff_data.get("speedup", "N/A")
        dtype = eff_data.get("dtype", "")

        # Prefer perf-stage errors: call/exe stderr often contains noisy warnings (e.g., PyTorch warnings
        # when numpy is missing). Prioritizing those can hide the real perf failure cause (e.g., empty perf json,
        # aggregation errors).
        error_msg = eff_data.get("error", "") or ""
        if not error_msg:
            error_msg = call_data.get("stderr", "") or exe_data.get("stderr", "")
        if not error_msg and correctness_ok and not perf_ok and perf_results_dir:
            op = os.path.splitext(os.path.basename(fname))[0]
            expected_perf_json = os.path.join(perf_results_dir, f"{op}.json")
            # First try to read actual error from perf_logs
            log_err = _read_perf_log_error(perf_results_dir, op)
            if log_err:
                error_msg = f"perf error: {log_err}"
            else:
                try:
                    if not os.path.exists(expected_perf_json):
                        error_msg = "missing perf json (no log error found)"
                    elif os.path.getsize(expected_perf_json) <= 2:
                        error_msg = "empty perf json"
                except Exception:
                    pass

        unified[fname] = {
            "file": fname,
            # --- primary status flags (single canonical set) ---
            "call_ok": compile_ok,
            "exe_ok": correctness_ok,
            "perf_ok": perf_ok,
            # --- performance numbers (shown when perf ran; N/A otherwise) ---
            "dtype": dtype,
            "runtime_ms": _na_if_needed(runtime_ms, should_na),
            "gbs": _na_if_needed(gbs, should_na),
            "tflops": _na_if_needed(tflops, should_na),
            "speedup": _na_if_needed(speedup, should_na),
            # --- code quality ---
            "maintainability": maintainability,
            "code_quality": code_quality,
            # --- per-size perf data & error detail ---
            "performance": perf if perf_ok else [],
            "error": error_msg,
            "code": call_data.get("code", ""),
        }

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(unified, f, indent=2)

    # Count pass/fail by stage
    total = len(unified)
    call_passed = sum(1 for v in unified.values() if v["call_ok"])
    exe_passed = sum(1 for v in unified.values() if v["call_ok"] and v["exe_ok"])
    perf_passed = sum(1 for v in unified.values() if v["call_ok"] and v["exe_ok"] and v["perf_ok"])
    
    # Global average speedup: sum(golden reference total ms) / sum(generated total ms)
    # Reference times must come from golden_results/*.json. If you mistakenly use perf_results for both
    # numerator and denominator (same source as runtime_ms), the ratio will be ~1.0 by construction.
    total_ref_ms_sum = 0.0
    total_gen_ms_sum = 0.0

    for fname, data in unified.items():
        if not data["perf_ok"] or not isinstance(data.get("runtime_ms"), (int, float)):
            continue
        ref_ms = golden_ref_ms.get(fname, 0.0)
        if ref_ms <= 0:
            continue
        total_ref_ms_sum += ref_ms
        total_gen_ms_sum += float(data["runtime_ms"])

    # Note: sum(ref)/sum(gen) is sensitive to extreme generated runtimes (e.g., a single anomalous timing row).
    avg_speedup_global = (
        total_ref_ms_sum / total_gen_ms_sum if total_gen_ms_sum > 0 else 0.0
    )
    # Per-task arithmetic mean: matches per-op speedup in 2_efficiency and is less dominated by extreme totals.
    speedups = []
    for fname, data in unified.items():
        if not data["perf_ok"]:
            continue
        sp = data.get("speedup")
        if isinstance(sp, (int, float)) and sp > 0:
            speedups.append(float(sp))
    avg_speedup_per_task = sum(speedups) / len(speedups) if speedups else 0.0

    # Write summary.json
    summary = {
        "method": os.path.basename(os.path.dirname(output_json)),
        "total": total,
        "call_pass": call_passed,
        "call_rate": round(call_passed / total * 100, 1) if total else 0.0,
        "exe_pass": exe_passed,
        "exe_rate": round(exe_passed / total * 100, 1) if total else 0.0,
        "perf_pass": perf_passed,
        "perf_rate": round(perf_passed / total * 100, 1) if total else 0.0,
        "avg_speedup": round(avg_speedup_global, 3),
        "avg_speedup_global": round(avg_speedup_global, 3),
        "avg_speedup_per_task": round(avg_speedup_per_task, 3),
    }
    
    summary_json = output_json.replace("metrics.json", "summary.json")
    with open(summary_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("Unified Results Summary:")
    print(f"{'='*60}")
    print(f"Total: {total}")
    print(f"Call Accuracy:  {call_passed}/{total} ({summary['call_rate']:.1f}%)")
    print(f"Exe Accuracy:   {exe_passed}/{total} ({summary['exe_rate']:.1f}%)")
    print(f"Perf Passed:    {perf_passed}/{total} ({summary['perf_rate']:.1f}%)")
    print(
        f"Avg Speedup (global sum_ref/sum_gen): {avg_speedup_global:.3f}x  "
        f"|  (per-task mean): {avg_speedup_per_task:.3f}x"
    )
    print(f"Output: {output_json}")
    print(f"Summary: {summary_json}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--call', type=str, default=None)
    parser.add_argument('--exe', type=str, default=None)
    parser.add_argument('--eff', type=str, default=None)
    parser.add_argument('--perf_dir', type=str, default=None)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    
    merge_results(args.call, args.exe, args.eff, args.perf_dir, args.output)
