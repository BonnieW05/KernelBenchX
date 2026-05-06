import torch
import torch.nn.functional as F

def gelu_bf16(input: torch.Tensor, approximate: str='none') -> torch.Tensor:
    """
    GELU activation with BF16 precision.
    Input and output must be torch.bfloat16.
    Implement using Triton kernel with BF16 I/O.
    """
    return F.gelu(input, approximate=approximate)

##################################################################################################################################################

import torch
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_tensor

def test_gelu_bf16():
    results = {}
    dtype = torch.bfloat16
    
    # Fixed test cases
    input_fixed = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device='cuda', dtype=dtype)
    out = gelu_bf16(input_fixed)
    results["test_fixed"] = out

    input_extreme = torch.tensor([-10.0, -3.0, -1.0, 0.0, 1.0, 3.0, 10.0], device='cuda', dtype=dtype)
    out = gelu_bf16(input_extreme)
    results["test_fixed_extreme"] = out

    input_2d = torch.tensor([[-2.0, -0.5, 0.0], [0.5, 2.0, 4.0]], device='cuda', dtype=dtype)
    out = gelu_bf16(input_2d)
    results["test_fixed_2d"] = out

    input_lin = torch.linspace(-6.0, 6.0, steps=257, device='cuda', dtype=dtype)
    out = gelu_bf16(input_lin)
    results["test_fixed_linspace"] = out

    out = gelu_bf16(input_extreme, approximate='tanh')
    results["test_fixed_approx_tanh"] = out

    for mode in ("standard", "outlier"):
        outs_none = []
        outs_tanh = []
        for _ in range(3):
            x = rand_tensor((2048,), dtype=dtype, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            outs_none.append(gelu_bf16(x, approximate='none'))
            outs_tanh.append(gelu_bf16(x, approximate='tanh'))
        results[f"test_random_{mode}_none"] = outs_none
        results[f"test_random_{mode}_tanh"] = outs_tanh
    
    return results

test_results = test_gelu_bf16()
