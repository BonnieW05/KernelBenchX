import torch
import torch.nn.functional as F

def layernorm_w8a8(input: torch.Tensor, normalized_shape, weight: torch.Tensor = None, bias: torch.Tensor = None, eps: float = 1e-5) -> torch.Tensor:
    """W8A8 layernorm benchmark (dynamic/online quantization).

    This is DYNAMIC quantization: your kernel receives fp32 inputs and must:
    1. Compute mean and variance in fp32 (MUST use fp32, NOT int32, to avoid catastrophic cancellation)
    2. Normalize using fp32 arithmetic with eps stabilization
    3. Apply affine transform (weight/bias) in fp32
    4. Output quantization (REQUIRED for W8A8 pipeline compatibility):
       - Compute per-row output scale: scale = max(abs(normalized)) / 127.0
       - Quantize to int8, then immediately dequantize to fp32
       - This allows downstream int8 operators to re-quantize with known scale

    Quantization scheme to implement:
    - Input: fp32 activations (received as-is).
    - Compute: Mean and variance MUST be computed in fp32 to avoid precision loss.
    - Internal representation can use int8 for memory efficiency, but stats must be fp32.
    - Output: fp32 (can internally quantize/dequantize if beneficial for perf).

    Accuracy requirement (against fp32 layer_norm): cosine_sim >= 0.95, l1_relative <= 0.05, rmse <= 0.1.
    """
    return F.layer_norm(input, normalized_shape, weight, bias, eps)

##################################################################################################################################################

import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
import torch
from data_utils import rand_tensor

def test_layernorm_w8a8():
    results = {}
    dtype = torch.float32

    # Corner case 1: constant input (zero variance; tests eps stability)
    input_const = torch.ones(2, 8, device='cuda', dtype=dtype) * 5.0
    results["corner_zero_variance"] = layernorm_w8a8(input_const, (8,), None, None, eps=1e-5)

    # Corner case 2: tiny eps (tests numerical stability / division-by-zero avoidance)
    input_tiny_var = torch.tensor([[1.0, 1.0 + 1e-7, 1.0, 1.0]], device='cuda', dtype=dtype)
    results["corner_tiny_eps"] = layernorm_w8a8(input_tiny_var, (4,), None, None, eps=1e-10)

    # Corner case 3: no affine parameters (weight=None, bias=None)
    input_no_affine = torch.randn(3, 16, device='cuda', dtype=dtype)
    results["corner_no_affine"] = layernorm_w8a8(input_no_affine, (16,), None, None)

    # Corner case 4: extreme weight/bias (tests affine transform dynamic range)
    input_norm = torch.randn(2, 4, device='cuda', dtype=dtype)
    weight_extreme = torch.tensor([100.0, 0.01, -100.0, 0.01], device='cuda', dtype=dtype)
    bias_extreme = torch.tensor([50.0, -50.0, 0.0, 100.0], device='cuda', dtype=dtype)
    results["corner_extreme_affine"] = layernorm_w8a8(input_norm, (4,), weight_extreme, bias_extreme)

    # Corner case 5: single-element normalized_shape
    input_d1 = torch.randn(4, 1, device='cuda', dtype=dtype)
    results["corner_single_dim"] = layernorm_w8a8(input_d1, (1,), None, None)

    for mode in ("standard", "outlier"):
        outs = []
        for B, D in ((32, 256), (16, 512)):
            x = rand_tensor((B, D), dtype=dtype, mode=mode, outlier_prob=0.01, outlier_scale=10.0).clamp(-10, 10)
            w = rand_tensor((D,), dtype=dtype, mode=mode, outlier_prob=0.01, outlier_scale=10.0).clamp(-10, 10)
            b = rand_tensor((D,), dtype=dtype, mode=mode, outlier_prob=0.01, outlier_scale=10.0).clamp(-10, 10)
            outs.append(layernorm_w8a8(x, (D,), w, b))
            outs.append(layernorm_w8a8(x, (D,), None, None))
        results[f"test_random_{mode}"] = outs
    
    return results

test_results = test_layernorm_w8a8()
