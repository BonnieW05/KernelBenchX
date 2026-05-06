import torch

def matmul_bf16(input: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    """
    Matrix multiplication with BF16 precision.
    Input and output must be torch.bfloat16.
    Implement using Triton kernel with BF16 I/O and FP32 accumulation.
    """
    return torch.matmul(input, other)

##################################################################################################################################################

import torch
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_tensor

def test_matmul_bf16():
    results = {}
    dtype = torch.bfloat16
    
    # 2D x 2D
    a = torch.randn(64, 128, device='cuda', dtype=dtype)
    b = torch.randn(128, 256, device='cuda', dtype=dtype)
    results["test_2d"] = matmul_bf16(a, b)
    
    # 3D batch matmul
    a = torch.randn(4, 64, 128, device='cuda', dtype=dtype)
    b = torch.randn(4, 128, 256, device='cuda', dtype=dtype)
    results["test_3d"] = matmul_bf16(a, b)
    
    # 1D x 2D
    a = torch.randn(128, device='cuda', dtype=dtype)
    b = torch.randn(128, 256, device='cuda', dtype=dtype)
    results["test_1d_2d"] = matmul_bf16(a, b)
    
    # Large matrix
    a = torch.randn(512, 1024, device='cuda', dtype=dtype)
    b = torch.randn(1024, 512, device='cuda', dtype=dtype)
    results["test_large"] = matmul_bf16(a, b)

    for mode in ("standard", "outlier"):
        outs = []
        a = rand_tensor((64, 128), dtype=dtype, mode=mode, outlier_prob=0.001, outlier_scale=10.0).clamp(-5.0, 5.0)
        b = rand_tensor((128, 64), dtype=dtype, mode=mode, outlier_prob=0.001, outlier_scale=10.0).clamp(-5.0, 5.0)
        outs.append(matmul_bf16(a, b))
        a = rand_tensor((4, 64, 128), dtype=dtype, mode=mode, outlier_prob=0.001, outlier_scale=10.0).clamp(-5.0, 5.0)
        b = rand_tensor((4, 128, 64), dtype=dtype, mode=mode, outlier_prob=0.001, outlier_scale=10.0).clamp(-5.0, 5.0)
        outs.append(matmul_bf16(a, b))
        results[f"test_random_{mode}"] = outs
    
    return results

test_results = test_matmul_bf16()
