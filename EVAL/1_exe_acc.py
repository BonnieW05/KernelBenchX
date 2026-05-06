import os
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
import re
import runpy
import random
from typing import Tuple
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../utils"))
from correction_utils import precision_metric, impl_must_export_kernel

try:
    from code_quality import analyze_code_quality
    QUALITY_AVAILABLE = True
except ImportError:
    QUALITY_AVAILABLE = False

_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.normpath(os.path.join(_EVAL_DIR, ".."))
_METADATA_PATH = os.path.join(_REPO_DIR, "data", "kernelbenchx_v1.json")

gold_folder = os.path.join(_REPO_DIR, "data", "kernelbenchx")


def _parse_gpus_arg(gpus_str: str):
    s = str(gpus_str).strip()
    if s.startswith('[') and s.endswith(']'):
        try:
            return [int(x) for x in re.findall(r"\d+", s)]
        except Exception:
            pass
    if "," in s:
        return [int(x.strip()) for x in s.split(",") if x.strip() != ""]
    return [int(s)]

def build_file_map():
    """Construct filename-to-category-path mapping table from metadata."""
    metadata = json.loads(open(_METADATA_PATH, 'r', encoding='utf-8').read())
    return {os.path.basename(item["file"]): item["file"] for item in metadata}

def load_precision_thresholds():
    """Load precision thresholds for each file from metadata."""
    metadata = json.loads(open(_METADATA_PATH, 'r', encoding='utf-8').read())
    thresholds = {}
    for item in metadata:
        filename = os.path.basename(item["file"])
        if "precision_thresholds" in item:
            thresholds[filename] = item["precision_thresholds"]
    return thresholds

def _get_tensor_stats(t, other=None):
    import torch
    n = t.numel()
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "has_nan": False, "has_inf": False}
    flat = t.detach().float().cpu().flatten()
    result = {
        "mean": flat.mean().item(),
        "std": flat.std().item() if n > 1 else 0.0,
        "min": flat.min().item(),
        "max": flat.max().item(),
        "has_nan": torch.isnan(t).any().item(),
        "has_inf": torch.isinf(t).any().item(),
    }
    if n < 1000:
        result["data"] = flat.tolist()
    else:
        k = 32
        result["head"] = flat[:k].tolist()
        result["tail"] = flat[-k:].tolist()
        if other is not None:
            other_flat = other.detach().float().cpu().flatten()
            if other_flat.numel() == n:
                top_idx = (flat - other_flat).abs().topk(k).indices
                result["max_error_indices"] = top_idx.tolist()
                result["max_error_llm"] = flat[top_idx].tolist()
                result["max_error_gold"] = other_flat[top_idx].tolist()
    return result

def _build_error_info(v1, v2, key_path, threshold_info, metrics):
    """Build detailed error information for logging."""
    error_info = {
        "key_path": key_path,
        "shape": tuple(v1.shape),
        "dtype": str(v1.dtype),
        "metrics": metrics,
        "llm_stats": _get_tensor_stats(v1, v2),
        "gold_stats": _get_tensor_stats(v2),
    }
    error_info.update(threshold_info)
    return error_info

def compare_python_files(file1, file2, precision_thresholds_map=None):
    """
    Compare the results of two Python files.

    The expected contract for benchmark test files is that they set a top-level
    variable named `test_results`, usually a dict whose values are torch.Tensors.

    To make random tests reproducible, we seed torch RNG before each file run.
    """
    error_details = []  # Collect detailed error information
    filename = os.path.basename(file1)
    custom_threshold = (precision_thresholds_map or {}).get(filename, None)
    
    def _seed_all(seed: int) -> None:
        import torch
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _default_tol(dtype) -> Tuple[float, float]:
        import torch
        if dtype in (torch.float16, torch.bfloat16):
            return 5e-3, 5e-3
        if dtype in (torch.float32, torch.float64):
            return 1e-5, 1e-5
        return 0.0, 0.0

    def _compare_values(v1, v2, key_path: str) -> None:
        import torch
        if isinstance(v1, torch.Tensor) and isinstance(v2, torch.Tensor):
            if v1.shape != v2.shape:
                raise AssertionError(f"shape mismatch at {key_path}: {tuple(v1.shape)} vs {tuple(v2.shape)}")
            if v1.dtype != v2.dtype:
                raise AssertionError(f"dtype mismatch at {key_path}: {v1.dtype} vs {v2.dtype}")
            
            # Use custom thresholds if provided; otherwise fall back to default tolerances.
            if custom_threshold:
                metrics = precision_metric(v1, v2)
                passed = (metrics["cosine_sim"] >= custom_threshold.get("cosine_sim", 0.99) and
                         metrics["l1_relative"] <= custom_threshold.get("l1_relative", 0.01) and
                         metrics["rmse"] <= custom_threshold.get("rmse", 0.01))
                if not passed:
                    error_details.append(_build_error_info(v1, v2, key_path, {"custom_threshold": custom_threshold}, metrics))
                    raise AssertionError(f"Precision threshold not met at {key_path}: {metrics}")
            else:
                rtol, atol = _default_tol(v1.dtype)
                try:
                    torch.testing.assert_close(v1, v2, rtol=rtol, atol=atol)
                except AssertionError as e:
                    metrics = precision_metric(v1, v2)
                    threshold_info = {"rtol": rtol, "atol": atol, "original_error": str(e)}
                    error_details.append(_build_error_info(v1, v2, key_path, threshold_info, metrics))
                    raise
            return
        if isinstance(v1, (list, tuple)) and isinstance(v2, (list, tuple)):
            if len(v1) != len(v2):
                raise AssertionError(f"length mismatch at {key_path}: {len(v1)} vs {len(v2)}")
            for i, (a, b) in enumerate(zip(v1, v2)):
                _compare_values(a, b, f"{key_path}[{i}]")
            return
        if isinstance(v1, dict) and isinstance(v2, dict):
            k1, k2 = set(v1.keys()), set(v2.keys())
            if k1 != k2:
                raise AssertionError(f"dict keys mismatch at {key_path}: {sorted(k1)} vs {sorted(k2)}")
            for k in sorted(k1):
                _compare_values(v1[k], v2[k], f"{key_path}.{k}")
            return

        if v1 != v2:
            raise AssertionError(f"value mismatch at {key_path}: {v1!r} vs {v2!r}")

    seed = int(os.environ.get("KERNELBENCHX_SEED", "0"))
    code = open(file1, 'r', encoding='utf-8').read()
    filename = file1.split("/")[-1]
    stem = os.path.splitext(filename)[0]
    impl = code.split("#" * 146)[0] if "#" * 146 in code else code
    ok_e, msg_e = impl_must_export_kernel(impl, stem)
    if not ok_e:
        print(f"[compare_python_files] FAIL {filename}: {msg_e}", flush=True)
        return False, filename, code, msg_e

    try:
        _seed_all(seed)
        g1 = runpy.run_path(file1, run_name="__main__")
        r1 = g1.get("test_results", None)

        _seed_all(seed)
        g2 = runpy.run_path(file2, run_name="__main__")
        r2 = g2.get("test_results", None)

        if r1 is None or r2 is None:
            raise AssertionError("missing `test_results` in one or both files")

        _compare_values(r1, r2, "test_results")
        return True, file1.split("/")[-1], code, ""
    except Exception as e:
        filename = file1.split("/")[-1]
        print(f"[compare_python_files] FAIL {filename}: {e}", flush=True)
        
        # Write detailed error logs
        if error_details:
            from datetime import datetime
            os.makedirs("test_failures", exist_ok=True)
            log_file = f"test_failures/{filename.replace('.py', '')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.json"
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump({"filename": filename, "llm_file": file1, "gold_file": file2, 
                          "seed": seed, "error_summary": str(e), "detailed_errors": error_details}, f, indent=2)
            print(f"  → Error log: {log_file}", flush=True)
            for err in error_details:
                m = err['metrics']
                print(f"  → {err['key_path']}: cosine={m['cosine_sim']:.6f}, l1={m['l1_relative']:.6f}, rmse={m['rmse']:.6f}", flush=True)
        
        return False, filename, code, str(e)

def test_close_parallel(llm_folder, gold_folder, gpus, delete=True, result_jsonl=None, analyze_quality=False):
    files = [f for f in os.listdir(llm_folder) if f.endswith(".py")]
    file_map = build_file_map()
    precision_thresholds_map = load_precision_thresholds()
    # Track correct executions
    correct_count = 0
    total_count = len(files)
    results = []

    with ProcessPoolExecutor(max_workers=len(gpus)) as executor:
        futures = []

        for idx, f in enumerate(files):
            # Convert to absolute paths to avoid issues when subprocesses chdir.
            file1 = os.path.abspath(os.path.join(llm_folder, f))
            file2 = os.path.abspath(os.path.join(gold_folder, file_map.get(f, f)))

            # Set GPU device for each task (distribute across GPUs)
            gpu_id = gpus[idx % len(gpus)]
            futures.append(executor.submit(run_with_gpu, file1, file2, gpu_id, precision_thresholds_map))

        # Process the results
        for future in futures:
            is_correct, file_name, code, error_msg = future.result()
            result = {"file": file_name, "ok": is_correct, "code": code, "stderr": error_msg}
            
            if is_correct and analyze_quality and QUALITY_AVAILABLE:
                file_path = os.path.join(llm_folder, file_name)
                result["quality"] = analyze_code_quality(file_path)
            
            results.append(result)

            if is_correct:
                correct_count += 1
            elif delete:
                file_path = os.path.join(llm_folder, file_name)
                os.remove(file_path)
                print(f"Deleted {file_name}", flush=True)

    # Calculate and print the correct execution rate
    if total_count > 0:
        correct_rate = (correct_count / total_count) * 100
    else:
        correct_rate = 0

    assert total_count == len(files), "error in files"
    print(f"\nCorrect execution rate: {correct_rate:.2f}% = {correct_count} / {total_count}", flush=True)
    
    if result_jsonl:
        os.makedirs(os.path.dirname(result_jsonl) or ".", exist_ok=True)
        with open(result_jsonl, 'w', encoding='utf-8') as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
    return results

def run_with_gpu(file1, file2, gpu_id, precision_thresholds_map=None):
    # Set the GPU device for this process
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    os.chdir(_REPO_DIR)
    
    # Compare the Python files
    return compare_python_files(file1, file2, precision_thresholds_map)  # Returns True/False

def execute_4folders(root_folder, gpus, result_jsonl=None, analyze_quality=False):
    for folder in os.listdir(root_folder):
        llm_folder = os.path.join(root_folder, folder)
        if not os.path.isdir(llm_folder):
            continue
        rj = result_jsonl.replace(".jsonl", f"_{folder}.jsonl") if result_jsonl else None
        test_close_parallel(llm_folder, gold_folder, gpus, result_jsonl=rj, analyze_quality=analyze_quality)
        print(f"above is the compare execution for {folder}", flush=True)
        print("========"*30, flush=True)

def execute_4folder(folder, gpus, result_jsonl=None, analyze_quality=False):
    assert os.path.isdir(folder), ""
    test_close_parallel(folder, gold_folder, gpus, result_jsonl=result_jsonl, analyze_quality=analyze_quality)
    print(f"above is the compare execution for {folder}", flush=True)
    print("========"*30, flush=True)


def main():
    parser = argparse.ArgumentParser(description="Call Triton-G operator.")
    parser.add_argument('--folder', type=str, required=True, help="root folder contains multiple test folders or just ont folder.")
    parser.add_argument('--GPUs', type=str, required=True, help="number of GPU available.")
    parser.add_argument('--result_jsonl', type=str, default=None, help="Path to output structured results.")
    parser.add_argument('--analyze_quality', action='store_true', help="Enable code quality analysis (requires radon).")
    
    args = parser.parse_args()
    assert os.path.isdir(args.folder), ""
    gpus = _parse_gpus_arg(args.GPUs)
    py_files = [f for f in os.listdir(args.folder) if f.endswith('.py')]
    if len(py_files) == 0:
        execute_4folders(args.folder, gpus, args.result_jsonl, args.analyze_quality)
    else:
        execute_4folder(args.folder, gpus, args.result_jsonl, args.analyze_quality)


if __name__ == "__main__":
    main()