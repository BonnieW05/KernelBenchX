import torch

def digamma(input_tensor):
    """
    Computes the digamma function (logarithmic derivative of the gamma function) for the input tensor.

    Args:
    - input_tensor (torch.Tensor): The tensor on which to compute the digamma function.

    Returns:
    - torch.Tensor: A tensor containing the digamma values.
    """
    return torch.special.digamma(input_tensor)

##################################################################################################################################################


import torch
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_tensor

# def digamma(input_tensor):
#     """
#     Computes the digamma function (logarithmic derivative of the gamma function) for the input tensor.

#     Args:
#     - input_tensor (torch.Tensor): The tensor on which to compute the digamma function.

#     Returns:
#     - torch.Tensor: A tensor containing the digamma values.
#     """
#     return torch.special.digamma(input_tensor)

def test_digamma():
    results = {}
    
    # Test case 1: Single positive value
    input_tensor = torch.tensor([1.0], device='cuda')
    results["test_case_1"] = digamma(input_tensor)
    
    # Test case 2: Single negative value
    input_tensor = torch.tensor([0.5], device='cuda')
    results["test_case_2"] = digamma(input_tensor)
    
    # Test case 3: Multiple positive values
    input_tensor = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    results["test_case_3"] = digamma(input_tensor)
    
    # Test case 4: Mixed positive and negative values
    input_tensor = torch.tensor([1.0, 0.5, 2.0, 1.5], device='cuda')
    results["test_case_4"] = digamma(input_tensor)

    for mode in ("standard", "outlier"):
        x = rand_tensor((1024,), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
        x = x.abs() + 0.1
        results[f"test_random_{mode}"] = digamma(x)
    
    return results

test_results = test_digamma()
