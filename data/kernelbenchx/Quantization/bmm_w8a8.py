import torch

def bmm_w8a8(input: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
    """W8A8 batch matmul benchmark (dynamic/online quantization).

    This is DYNAMIC quantization: your kernel receives fp32 inputs and must:
    1. Compute quantization scales at runtime (e.g., max(abs(input))/127 for symmetric)
    2. Quantize both operands to int8
    3. Perform int32 accumulation
    4. Dequantize back to fp32 output

    Quantization scheme to implement:
    - Symmetric quantization is recommended (simpler and faster than asymmetric).
    - Activations: per-(B,M) row-wise scale or per-tensor per-batch.
    - Weights/second operand: per-(B,N) column-wise scale, optionally group-wise along K.
    - Scales computed at runtime inside the kernel.

    Accuracy requirement (against fp32 bmm): cosine_sim >= 0.95, l1_relative <= 0.05, rmse <= 0.1.
    """
    return torch.bmm(input, mat2)

##################################################################################################################################################

import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
import torch
from data_utils import rand_tensor

def test_bmm_w8a8():
    results = {}
    dtype = torch.float32

    # Corner case 1: batch=1 (degenerates to a single matmul)
    input_b1 = torch.randn(1, 8, 16, device='cuda', dtype=dtype)
    mat2_b1 = torch.randn(1, 16, 8, device='cuda', dtype=dtype)
    results["corner_batch1"] = bmm_w8a8(input_b1, mat2_b1)

    # Corner case 2: extreme shapes (narrow matrices: M=1 or N=1)
    input_narrow = torch.randn(2, 1, 32, device='cuda', dtype=dtype)  # M=1
    mat2_narrow = torch.randn(2, 32, 1, device='cuda', dtype=dtype)   # N=1
    results["corner_narrow"] = bmm_w8a8(input_narrow, mat2_narrow)

    # Corner case 3: one all-zero batch (tests per-batch scale handling)
    input_zerobatch = torch.randn(3, 4, 8, device='cuda', dtype=dtype)
    input_zerobatch[1, :, :] = 0.0  # batch index 1 is all zeros
    mat2_norm = torch.randn(3, 8, 4, device='cuda', dtype=dtype)
    results["corner_zero_batch"] = bmm_w8a8(input_zerobatch, mat2_norm)

    # Corner case 4: asymmetric dynamic ranges (input range >> mat2 range)
    input_large = torch.randn(2, 8, 16, device='cuda', dtype=dtype) * 50
    mat2_small = torch.randn(2, 16, 8, device='cuda', dtype=dtype) * 0.1
    results["corner_asymmetric_scale"] = bmm_w8a8(input_large, mat2_small)

    for mode in ("standard", "outlier"):
        outs = []
        for B, M, K, N in ((2, 32, 64, 48), (1, 16, 32, 16)):
            x = rand_tensor((B, M, K), dtype=dtype, mode=mode, outlier_prob=0.01, outlier_scale=10.0).clamp(-10, 10)
            w = rand_tensor((B, K, N), dtype=dtype, mode=mode, outlier_prob=0.01, outlier_scale=10.0).clamp(-10, 10)
            outs.append(bmm_w8a8(x, w))
        results[f"test_random_{mode}"] = outs
    
    return results

test_results = test_bmm_w8a8()
