import torch
import torch.nn.functional as F

def gelu_int8(input: torch.Tensor, approximate: str='none') -> torch.Tensor:
    """
    GELU activation with INT8 precision.
    Input and output must be torch.int8.
    Implement using Triton kernel with INT8 I/O.
    """
    return F.gelu(input.float(), approximate=approximate).to(torch.int8)

##################################################################################################################################################

import torch
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_int

def test_gelu_int8():
    results = {}
    dtype = torch.int8
    
    # Fixed test vectors
    input_fixed = torch.tensor([-10, -5, 0, 5, 10], device='cuda', dtype=dtype)
    results["test_fixed"] = gelu_int8(input_fixed)
    
    # int8 boundary values
    input_edges = torch.tensor([-128, -127, -1, 0, 1, 126, 127], device='cuda', dtype=dtype)
    results["test_edges"] = gelu_int8(input_edges)
    
    # 2D test
    input_2d = torch.tensor([[-10, -5, 0], [5, 10, 20]], device='cuda', dtype=dtype)
    results["test_2d"] = gelu_int8(input_2d)
    
    # approximate='tanh'
    results["test_approx_tanh"] = gelu_int8(input_fixed, approximate='tanh')

    for mode in ("standard", "outlier"):
        outs_none = []
        outs_tanh = []
        for _ in range(3):
            if mode == "standard":
                x = rand_int((4096,), low=-20, high=20, device="cuda", dtype=torch.int16).to(torch.int8)
            else:
                x = rand_int((4096,), low=-128, high=128, device="cuda", dtype=torch.int16).to(torch.int8)
            outs_none.append(gelu_int8(x, approximate='none'))
            outs_tanh.append(gelu_int8(x, approximate='tanh'))
        results[f"test_random_{mode}_none"] = outs_none
        results[f"test_random_{mode}_tanh"] = outs_tanh
    
    return results

test_results = test_gelu_int8()
