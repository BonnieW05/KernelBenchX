import torch

def permute_copy(input: torch.Tensor, dims: list) -> torch.Tensor:
    """
    Performs the same operation as torch.permute, which rearranges the dimensions
    of the input tensor according to the specified dims, but all output tensors
    are freshly created instead of aliasing the input.
    
    Args:
        input (torch.Tensor): The input tensor to be permuted.
        dims (list): List of integers representing the target order of dimensions.
        
    Returns:
        torch.Tensor: The new tensor with the dimensions permuted.
    """
    return input.permute(dims).clone()

##################################################################################################################################################


import torch
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_tensor

def test_permute_copy():
    results = {}

    # Test case 1: Simple 2D tensor permutation
    tensor_2d = torch.tensor([[1, 2], [3, 4]], device='cuda')
    results["test_case_1"] = permute_copy(tensor_2d, [1, 0])

    # Test case 2: 3D tensor permutation
    tensor_3d = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], device='cuda')
    results["test_case_2"] = permute_copy(tensor_3d, [2, 0, 1])

    # Test case 3: Permutation with no change
    tensor_no_change = torch.tensor([1, 2, 3, 4], device='cuda')
    results["test_case_3"] = permute_copy(tensor_no_change, [0])

    # Test case 4: Higher dimensional tensor permutation
    tensor_4d = torch.rand((2, 3, 4, 5), device='cuda')
    results["test_case_4"] = permute_copy(tensor_4d, [3, 2, 1, 0])

    for mode in ("standard", "outlier"):
        outs = []
        for _ in range(3):
            x = rand_tensor((4, 8, 16, 32), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=20.0)
            outs.append(permute_copy(x, [3, 0, 2, 1]))
        results[f"test_random_{mode}"] = outs

    return results

test_results = test_permute_copy()
