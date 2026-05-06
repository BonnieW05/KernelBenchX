import torch
import torch.nn.functional as F

def relu_bf16(input: torch.Tensor) -> torch.Tensor:
    """
    ReLU activation with BF16 precision.
    Input and output must be torch.bfloat16.
    Implement using Triton kernel with BF16 I/O.
    """
    return F.relu(input)

##################################################################################################################################################

import torch
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_tensor

def test_relu_bf16():
    results = {}
    dtype = torch.bfloat16
    
    input_fixed = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device='cuda', dtype=dtype)
    out = relu_bf16(input_fixed)
    results["test_fixed"] = out

    input_2d = torch.tensor([[-3.0, -0.0, 0.0, 1.0], [2.0, -2.0, 3.0, -4.0]], device='cuda', dtype=dtype)
    out = relu_bf16(input_2d)
    results["test_fixed_2d"] = out

    input_extreme = torch.tensor([-1e4, -100.0, -1.0, 0.0, 1.0, 100.0, 1e4], device='cuda', dtype=dtype)
    out = relu_bf16(input_extreme)
    results["test_fixed_extreme"] = out

    input_zeros = torch.zeros((4, 4), device='cuda', dtype=dtype)
    out = relu_bf16(input_zeros)
    results["test_fixed_zeros"] = out

    input_pattern = torch.tensor([-1.0, 1.0, -1.0, 1.0, 0.0, 0.0], device='cuda', dtype=dtype)
    out = relu_bf16(input_pattern)
    results["test_fixed_pattern"] = out

    for mode in ("standard", "outlier"):
        outs = []
        for _ in range(3):
            x = rand_tensor((1024, 1024), dtype=dtype, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            outs.append(relu_bf16(x))
        results[f"test_random_{mode}"] = outs
    
    return results

test_results = test_relu_bf16()
