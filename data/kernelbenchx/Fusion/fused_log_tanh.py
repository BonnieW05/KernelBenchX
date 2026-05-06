import torch
import torch.nn.functional as F

def fused_log_tanh(input, out=None):
    if torch.any(input <= 0):
        raise ValueError('All input elements must be positive for the logarithm function to be defined.')
    result = torch.tanh(torch.log(input))
    if out is not None:
        out.copy_(result)
        return out
    return result

##################################################################################################################################################


import torch
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_tensor

# def log_tanh(input, out=None):
#     if torch.any(input <= 0):
#         raise ValueError('All input elements must be positive for the logarithm function to be defined.')
#     result = torch.tanh(torch.log(input))
#     if out is not None:
#         out.copy_(result)
#         return out
#     return result

def test_log_tanh():
    results = {}
    
    # Test case 1: Basic functionality with positive values
    input1 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    results["test_case_1"] = fused_log_tanh(input1)
    
    # Test case 2: Check behavior with out parameter
    input2 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    out2 = torch.empty(3, device='cuda')
    fused_log_tanh(input2, out=out2)
    results["test_case_2"] = out2
    
    # Test case 3: Edge case with values close to zero but positive
    input3 = torch.tensor([0.1, 0.01, 0.001], device='cuda')
    results["test_case_3"] = fused_log_tanh(input3)
    
    # Test case 4: Exception handling with non-positive values
    try:
        input4 = torch.tensor([-1.0, 0.0, 2.0], device='cuda')
        fused_log_tanh(input4)
    except ValueError as e:
        results["test_case_4"] = str(e)

    for mode in ("standard", "outlier"):
        outs = []
        for _ in range(3):
            x = rand_tensor((4096,), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            x = x.abs() + 0.1
            outs.append(fused_log_tanh(x))
        results[f"test_random_{mode}"] = outs
    
    return results

test_results = test_log_tanh()
