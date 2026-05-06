import torch


def sqrt_tanh(input, out=None):
    if out is None:
        out = torch.empty_like(input)
    out = torch.sqrt(input)
    out[input < 0] = float('nan')
    out = torch.tanh(out)
    return out

##################################################################################################################################################


import torch
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_tensor

def test_sqrt_tanh():
    results = {}

    # Test case 1: Positive values
    input1 = torch.tensor([4.0, 9.0, 16.0], device='cuda')
    results["test_case_1"] = sqrt_tanh(input1)

    # Test case 2: Negative values
    input2 = torch.tensor([-4.0, -9.0, -16.0], device='cuda')
    results["test_case_2"] = sqrt_tanh(input2)

    # Test case 3: Mixed values
    input3 = torch.tensor([4.0, -9.0, 16.0, -1.0], device='cuda')
    results["test_case_3"] = sqrt_tanh(input3)

    # Test case 4: Zero values
    input4 = torch.tensor([0.0, 0.0, 0.0], device='cuda')
    results["test_case_4"] = sqrt_tanh(input4)

    for mode in ("standard", "outlier"):
        x = rand_tensor((64, 64), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
        x = x.abs()
        results[f"test_random_{mode}"] = sqrt_tanh(x)

    return results

test_results = test_sqrt_tanh()
