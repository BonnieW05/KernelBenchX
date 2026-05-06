import torch

def min(input_tensor, dim, keepdim=False):
    """
    Returns the minimum value of each row (or the specified dimension) of the input tensor 
    in the given dimension, along with the index location of each minimum value found.

    Args:
        input_tensor (Tensor): The input tensor.
        dim (int): The dimension to reduce.
        keepdim (bool): Whether the output tensor has `dim` retained or not.

    Returns:
        tuple: A tuple of two tensors - 
               1. The minimum values for each row (or dimension).
               2. The indices of the minimum values in that dimension.
    """
    (values, indices) = torch.min(input_tensor, dim, keepdim)
    return (values, indices)

##################################################################################################################################################


import torch
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_tensor

def test_min():
    results = {}

    # Test case 1: 2D tensor, dim=0, keepdim=False
    input_tensor = torch.tensor([[1, 2, 3], [4, 0, 6]], device='cuda')
    results["test_case_1"] = min(input_tensor, dim=0)

    # Test case 2: 2D tensor, dim=1, keepdim=False
    input_tensor = torch.tensor([[1, 2, 3], [4, 0, 6]], device='cuda')
    results["test_case_2"] = min(input_tensor, dim=1)

    # Test case 3: 3D tensor, dim=2, keepdim=True
    input_tensor = torch.tensor([[[1, 2, 3], [4, 0, 6]], [[7, 8, 9], [10, 11, 12]]], device='cuda')
    results["test_case_3"] = min(input_tensor, dim=2, keepdim=True)

    # Test case 4: 1D tensor, dim=0, keepdim=False
    input_tensor = torch.tensor([1, 2, 3, 0, 4, 5], device='cuda')
    results["test_case_4"] = min(input_tensor, dim=0)

    for mode in ("standard", "outlier"):
        outs = []
        x2 = rand_tensor((128, 256), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
        outs.append(min(x2, dim=0))
        outs.append(min(x2, dim=1))
        x3 = rand_tensor((8, 16, 32), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
        outs.append(min(x3, dim=2, keepdim=True))
        results[f"test_random_{mode}"] = outs

    return results

test_results = test_min()
