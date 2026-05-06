import torch


def reciprocal(input, out=None):
    """
    Computes the reciprocal of each element in the input tensor.
    
    Args:
        input (Tensor): The input tensor whose reciprocal is computed.
        out (Tensor, optional): The output tensor. If provided, the result will be stored in it.
    
    Returns:
        Tensor: A new tensor containing the reciprocal of each element in the input tensor.
    
    Example:
        >>> import torch
        >>> a = torch.randn(4)
        >>> a
        tensor([-0.4595, -2.1219, -1.4314,  0.7298])
        >>> reciprocal(a)
        tensor([-2.1763, -0.4713, -0.6986,  1.3702])
    """
    return torch.reciprocal(input, out=out)

##################################################################################################################################################


import torch
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_tensor

def test_reciprocal():
    results = {}

    # Test case 1: Basic test with positive and negative numbers
    a = torch.tensor([-0.4595, -2.1219, -1.4314, 0.7298], device='cuda')
    results["test_case_1"] = reciprocal(a)

    # Test case 2: Test with a tensor containing zero (expecting inf)
    b = torch.tensor([0.0, 1.0, -1.0, 2.0], device='cuda')
    results["test_case_2"] = reciprocal(b)

    # Test case 3: Test with a tensor containing large numbers
    c = torch.tensor([1e10, -1e10, 1e-10, -1e-10], device='cuda')
    results["test_case_3"] = reciprocal(c)

    # Test case 4: Test with a tensor of ones (expecting ones)
    d = torch.ones(4, device='cuda')
    results["test_case_4"] = reciprocal(d)

    for mode in ("standard", "outlier"):
        x = rand_tensor((64, 64), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
        x = x.sign() * x.abs().clamp_min(1e-3)
        results[f"test_random_{mode}"] = reciprocal(x)

    return results

test_results = test_reciprocal()
