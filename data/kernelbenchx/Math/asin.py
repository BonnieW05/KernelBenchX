import torch

def asin(input_tensor):
    """
    Compute the elementwise arcsine (asin).

    Args:
        input_tensor (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Elementwise asin(input_tensor). Values outside [-1, 1] produce NaNs.
    """
    if not isinstance(input_tensor, torch.Tensor):
        raise ValueError('The input must be a torch.Tensor.')
    return torch.asin(input_tensor)

##################################################################################################################################################


import torch
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_tensor

def test_asin():
    results = {}

    # Test case 1: Valid input within range [-1, 1]
    input_tensor_1 = torch.tensor([0.0, 0.5, -0.5, 1.0, -1.0], device='cuda')
    results["test_case_1"] = asin(input_tensor_1)

    # Test case 2: Input values exceeding the range [-1, 1]
    input_tensor_2 = torch.tensor([0.9, -0.9], device='cuda')
    results["test_case_2"] = asin(input_tensor_2)

    # Test case 3: Empty tensor
    input_tensor_3 = torch.tensor([], device='cuda')
    results["test_case_3"] = asin(input_tensor_3)

    # Test case 4: Single element tensor
    input_tensor_4 = torch.tensor([0.707], device='cuda')
    results["test_case_4"] = asin(input_tensor_4)

    for mode in ("standard", "outlier"):
        outs = []
        for shape in ((1024,), (64, 64)):
            x = rand_tensor(shape, dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            x = x.tanh()
            outs.append(asin(x))
        results[f"test_random_{mode}"] = outs

    return results

test_results = test_asin()
