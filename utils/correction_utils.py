import ast
import torch
import json
import os
from datetime import datetime
from typing import Tuple


def precision_metric(output, reference, round_num=None):
    if output.numel() > 200000:
        output, reference = output.cpu(), reference.cpu()

    x = output.float()
    ref = reference.float()

    cosine = torch.nn.functional.cosine_similarity(x.reshape(1, -1), ref.reshape(1, -1)).item()
    denom = torch.abs(ref).sum().clamp_min(1e-12)
    l1_rel = (torch.abs(x - ref).sum() / denom).item()
    rmse = torch.sqrt(torch.mean((x - ref) ** 2)).item()

    if round_num is not None:
        cosine = round(cosine, round_num)
        l1_rel = round(l1_rel, round_num)
        rmse = round(rmse, round_num)

    return {"cosine_sim": cosine, "l1_relative": l1_rel, "rmse": rmse}


def check_precision(output, reference, thresholds=None, round_num=None):
    thresholds = thresholds or {"cosine_sim": 0.99, "l1_relative": 0.01, "rmse": 0.01}
    metrics = precision_metric(output, reference, round_num=round_num)
    passed = (
        metrics["cosine_sim"] >= thresholds.get("cosine_sim", 0.99)
        and metrics["l1_relative"] <= thresholds.get("l1_relative", 0.01)
        and metrics["rmse"] <= thresholds.get("rmse", 0.01)
    )
    return passed, metrics


def log_test_failure(test_name, test_case, params, output, reference, metrics, thresholds=None, log_dir="./test_failures"):
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{test_name}_{test_case}_{timestamp}.json")
    
    failure_info = {
        "test_name": test_name,
        "test_case": test_case,
        "timestamp": timestamp,
        "params": params,
        "actual_metrics": metrics,
        "required_thresholds": thresholds or {},
        "output_shape": list(output.shape) if hasattr(output, 'shape') else None,
        "output_dtype": str(output.dtype) if hasattr(output, 'dtype') else None,
        "reference_shape": list(reference.shape) if hasattr(reference, 'shape') else None,
        "reference_dtype": str(reference.dtype) if hasattr(reference, 'dtype') else None,
    }
    
    with open(log_file, 'w') as f:
        json.dump(failure_info, f, indent=2)
    
    return log_file


def impl_must_export_kernel(impl_code: str, stem: str) -> Tuple[bool, str]:
    """The implementation segment must export `def <stem>(...)` or `kernel_function(...)` (or assign to them).

    This prevents accidental passing by calling a built-in fallback/reference implementation.
    """
    impl_code = (impl_code or "").strip()
    if not impl_code:
        return False, "kernel entry check: empty implementation segment"
    try:
        tree = ast.parse(impl_code)
    except SyntaxError as e:
        return False, f"kernel entry check: syntax error in implementation: {e}"
    names = {stem, "kernel_function"}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            return True, ""
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in names:
                    return True, ""
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id in names:
            return True, ""
    return (
        False,
        f"kernel entry check: need `def {stem}(...)` or `def kernel_function(...)` "
        f"(or assign `{stem}` / `kernel_function`).",
    )
