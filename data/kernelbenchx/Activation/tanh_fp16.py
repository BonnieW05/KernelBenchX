import torch

def tanh_fp16(input: torch.Tensor) -> torch.Tensor:
    """
    Tanh activation with FP16 precision.
    Input and output must be torch.float16.
    Implement using Triton kernel with FP16 I/O.
    """
    return torch.tanh(input)

##################################################################################################################################################

import torch
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_tensor

def test_tanh_fp16():
    results = {}
    dtype = torch.float16
    
    input_fixed = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device='cuda', dtype=dtype)
    out = tanh_fp16(input_fixed)
    results["test_fixed"] = out

    input_extreme = torch.tensor([-20.0, -10.0, -3.0, -1.0, 0.0, 1.0, 3.0, 10.0, 20.0], device='cuda', dtype=dtype)
    out = tanh_fp16(input_extreme)
    results["test_fixed_extreme"] = out

    input_2d = torch.tensor([[-6.0, -2.0, 0.0], [2.0, 6.0, 12.0]], device='cuda', dtype=dtype)
    out = tanh_fp16(input_2d)
    results["test_fixed_2d"] = out

    input_lin = torch.linspace(-12.0, 12.0, steps=257, device='cuda', dtype=dtype)
    out = tanh_fp16(input_lin)
    results["test_fixed_linspace"] = out

    for mode in ("standard", "outlier"):
        outs = []
        for _ in range(3):
            x = rand_tensor((1024, 1024), dtype=dtype, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            outs.append(tanh_fp16(x))
        results[f"test_random_{mode}"] = outs
    
    return results

test_results = test_tanh_fp16()
