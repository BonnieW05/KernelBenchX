import torch

def erf(input_tensor):
    """
    Compute the elementwise error function (erf).

    Args:
        input_tensor (Tensor): Input tensor.

    Returns:
        Tensor: Elementwise erf(input_tensor).
    """
    return torch.special.erf(input_tensor)

##################################################################################################################################################


import torch
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_tensor

# def erf(input_tensor):
#     """
#     Compute the elementwise error function (erf).

#     Args:
#         input_tensor (Tensor): Input tensor.

#     Returns:
#         Tensor: Elementwise erf(input_tensor).
#     """
#     return torch.special.erf(input_tensor)

def test_erf():
    results = {}
    
    # Test case 1: Single element tensor
    input_tensor = torch.tensor([0.5], device='cuda')
    results["test_case_1"] = erf(input_tensor)
    
    # Test case 2: Multi-element tensor
    input_tensor = torch.tensor([0.5, -1.0, 2.0], device='cuda')
    results["test_case_2"] = erf(input_tensor)
    
    # Test case 3: Large values tensor
    input_tensor = torch.tensor([10.0, -10.0], device='cuda')
    results["test_case_3"] = erf(input_tensor)
    
    # Test case 4: Zero tensor
    input_tensor = torch.tensor([0.0], device='cuda')
    results["test_case_4"] = erf(input_tensor)

    for mode in ("standard", "outlier"):
        outs = []
        for shape in ((1024,), (64, 64)):
            x = rand_tensor(shape, dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            x = x.clamp(-10.0, 10.0)
            outs.append(erf(x))
        results[f"test_random_{mode}"] = outs
    
    return results

test_results = test_erf()
