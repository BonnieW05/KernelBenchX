import torch

def mul_bf16(input: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    """
    Element-wise multiplication with BF16 precision.
    Input and output must be torch.bfloat16.
    Implement using Triton kernel with BF16 I/O.
    """
    return torch.mul(input, other)

##################################################################################################################################################

import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_tensor

def test_mul_bf16():
    results = {}
    dtype = torch.bfloat16
    
    input_fixed = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda', dtype=dtype)
    other_fixed = torch.tensor([[0.5, 1.5], [2.5, 3.5]], device='cuda', dtype=dtype)
    results["test_fixed"] = mul_bf16(input_fixed, other_fixed)

    input_neg = torch.tensor([[-1.0, 0.0], [2.0, -3.0]], device='cuda', dtype=dtype)
    other_neg = torch.tensor([[4.0, -5.0], [-6.0, 7.0]], device='cuda', dtype=dtype)
    results["test_fixed_negative"] = mul_bf16(input_neg, other_neg)

    input_zeros = torch.zeros((2, 2), device='cuda', dtype=dtype)
    results["test_fixed_zeros"] = mul_bf16(input_zeros, other_fixed)

    input_b = torch.tensor([[1.0, 2.0, 3.0]], device='cuda', dtype=dtype)
    other_b = torch.tensor([10.0, 0.0, -10.0], device='cuda', dtype=dtype)
    results["test_fixed_broadcast"] = mul_bf16(input_b, other_b)

    for mode in ("standard", "outlier"):
        outs = []
        for _ in range(3):
            a = rand_tensor((1024, 1024), dtype=dtype, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            b = rand_tensor((1024, 1024), dtype=dtype, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            outs.append(mul_bf16(a, b))
        results[f"test_random_{mode}"] = outs
    
    return results

test_results = test_mul_bf16()
