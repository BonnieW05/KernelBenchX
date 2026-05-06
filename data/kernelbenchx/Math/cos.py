import torch

def cos(input_tensor):
    """
    Compute the elementwise cosine (cos).

    Args:
        input_tensor (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Elementwise cos(input_tensor).
    """
    return torch.cos(input_tensor)

##################################################################################################################################################


import torch
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_tensor

def test_cos():
    results = {}

    # Test case 1: Single positive value
    input_tensor_1 = torch.tensor([0.0], device='cuda')
    results["test_case_1"] = cos(input_tensor_1)

    # Test case 2: Single negative value
    input_tensor_2 = torch.tensor([-3.14159265 / 2], device='cuda')
    results["test_case_2"] = cos(input_tensor_2)

    # Test case 3: Multiple values
    input_tensor_3 = torch.tensor([0.0, 3.14159265 / 2, 3.14159265], device='cuda')
    results["test_case_3"] = cos(input_tensor_3)

    # Test case 4: Large tensor
    input_tensor_4 = torch.linspace(-3.14159265, 3.14159265, steps=1000, device='cuda')
    results["test_case_4"] = cos(input_tensor_4)

    for mode in ("standard", "outlier"):
        outs = []
        for shape in ((1024,), (64, 64)):
            x = rand_tensor(shape, dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            outs.append(cos(x))
        results[f"test_random_{mode}"] = outs

    return results

test_results = test_cos()
