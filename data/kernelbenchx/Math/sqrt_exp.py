import torch


def sqrt_exp(input, out=None):
    """
    Computes the square root of each element in :attr:`input`, 
    and then applies the exponential function to the square-rooted values.
    
    Args:
        input (Tensor): The input tensor.
        out (Tensor, optional): The output tensor.
    
    Returns:
        Tensor: A tensor containing e^(sqrt(input_i)) for each element in input.
    
    Example:
        >>> import torch
        >>> a = torch.tensor([0.25, 1.0, 4.0, 9.0])
        >>> result = sqrt_exp(a)
        >>> print(result)
        tensor([ 1.2840,  2.7183,  7.3891, 20.0855])
    """
    if out is None:
        out = torch.exp(torch.sqrt(input))
    else:
        torch.sqrt(input, out=out)
        torch.exp(out, out=out)
    return out

##################################################################################################################################################


import torch
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_tensor

def test_sqrt_exp():
    results = {}

    # Test case 1: Basic functionality with GPU tensor
    a = torch.tensor([0.25, 1.0, 4.0, 9.0], device='cuda')
    results["test_case_1"] = sqrt_exp(a)

    # Test case 2: Empty tensor
    b = torch.tensor([], device='cuda')
    results["test_case_2"] = sqrt_exp(b)

    # Test case 3: Tensor with zero values
    c = torch.tensor([0.0, 0.0, 0.0], device='cuda')
    results["test_case_3"] = sqrt_exp(c)

    # Test case 4: Using the out parameter
    d = torch.tensor([0.25, 1.0, 4.0, 9.0], device='cuda')
    out_tensor = torch.empty_like(d)
    results["test_case_4"] = sqrt_exp(d, out=out_tensor)

    for mode in ("standard", "outlier"):
        outs = []
        x = rand_tensor((64, 64), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
        x = x.abs().clamp_max(100.0)
        outs.append(sqrt_exp(x))
        out = torch.empty_like(x)
        outs.append(sqrt_exp(x, out=out))
        results[f"test_random_{mode}"] = outs

    return results

test_results = test_sqrt_exp()
