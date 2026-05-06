import torch

def matmul_w8a8(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """W8A8 matmul benchmark (dynamic/online quantization).

    This is DYNAMIC quantization: your kernel receives fp32/fp16 inputs and must:
    1. Compute quantization scales (e.g., max(abs(input))/127 for symmetric)
    2. Quantize activations and weights to int8
    3. Perform int32 accumulation
    4. Dequantize back to fp32 output

    Quantization scheme to implement:
    - Activations: per-tensor or per-row (M dimension) symmetric quantization.
    - Weights: per-column (N dimension) symmetric quantization.
    - Scales are computed at runtime inside the kernel.
    - Optional: group-wise scales (e.g., group size 64 along K) to reduce error.

    Accuracy requirement (against fp32 matmul): cosine_sim >= 0.95, l1_relative <= 0.05, rmse <= 0.1.
    """
    return torch.matmul(input, weight)

##################################################################################################################################################

import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
import torch
from data_utils import rand_tensor

def test_matmul_w8a8():
    results = {}
    dtype = torch.float32

    # Corner case 1: all-zero column (per-column scale boundary: scale=0)
    weight_zero_col = torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, 3.0]], device='cuda', dtype=dtype)
    input_zc = torch.tensor([[1.0, 2.0]], device='cuda', dtype=dtype)
    results["corner_zero_column"] = matmul_w8a8(input_zc, weight_zero_col)

    # Corner case 2: single-element matmul (1x1 @ 1x1)
    results["corner_1x1"] = matmul_w8a8(
        torch.tensor([[127.0]], device='cuda', dtype=dtype),
        torch.tensor([[-127.0]], device='cuda', dtype=dtype)
    )

    # Corner case 3: extreme scale differences (one huge value, others tiny; tests quantization error)
    weight_extreme = torch.tensor([[100.0, 0.01, -0.01], [0.01, -100.0, 0.01]], device='cuda', dtype=dtype)
    input_ex = torch.tensor([[1.0, 1.0]], device='cuda', dtype=dtype)
    results["corner_extreme_scale"] = matmul_w8a8(input_ex, weight_extreme)

    # Corner case 4: negative zero edge case (-0.0 vs 0.0)
    weight_negzero = torch.tensor([[0.0, -0.0], [-0.0, 1.0]], device='cuda', dtype=dtype)
    input_nz = torch.tensor([[1.0, -1.0]], device='cuda', dtype=dtype)
    results["corner_neg_zero"] = matmul_w8a8(input_nz, weight_negzero)

    # Random tests (standard + outlier)
    for mode in ("standard", "outlier"):
        outs = []
        for M, K, N in ((32, 256, 128), (64, 128, 64)):
            x = rand_tensor((M, K), dtype=dtype, mode=mode, outlier_prob=0.01, outlier_scale=10.0).clamp(-10, 10)
            w = rand_tensor((K, N), dtype=dtype, mode=mode, outlier_prob=0.01, outlier_scale=10.0).clamp(-10, 10)
            outs.append(matmul_w8a8(x, w))
        results[f"random_{mode}"] = outs
    
    return results

test_results = test_matmul_w8a8()
