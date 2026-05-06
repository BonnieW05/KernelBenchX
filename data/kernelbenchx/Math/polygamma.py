import torch

def polygamma(n, input, *, out=None):
    """
    Computes the n-th derivative of the digamma function on input.
    
    Args:
        n (int): The order of the polygamma function (nonnegative integer).
        input (Tensor): The input tensor (values at which to evaluate the function).
        out (Tensor, optional): A tensor to store the result.

    Returns:
        Tensor: The result of the n-th derivative of the digamma function.
    
    Example:
        >>> a = torch.tensor([1, 0.5])
        >>> polygamma(1, a)
        tensor([1.64493, 4.9348])
        >>> polygamma(2, a)
        tensor([ -2.4041, -16.8288])
        >>> polygamma(3, a)
        tensor([ 6.4939, 97.4091])
        >>> polygamma(4, a)
        tensor([ -24.8863, -771.4742])
    """
    input = torch.as_tensor(input)
    if not isinstance(n, int) or n < 0:
        raise ValueError('n must be a non-negative integer.')
    result = torch.special.polygamma(n, input)
    if out is not None:
        out.copy_(result)
    return result

##################################################################################################################################################


import torch
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_tensor

def test_polygamma():
    results = {}

    # Test case 1: Basic functionality with n=1
    a = torch.tensor([1, 0.5], device='cuda')
    results["test_case_1"] = polygamma(1, a)

    # Test case 2: Basic functionality with n=2
    results["test_case_2"] = polygamma(2, a)

    # Test case 3: Basic functionality with n=3
    results["test_case_3"] = polygamma(3, a)

    # Test case 4: Basic functionality with n=4
    results["test_case_4"] = polygamma(4, a)

    for mode in ("standard", "outlier"):
        outs = []
        x = rand_tensor((1024,), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
        x = x.abs() + 0.1
        outs.append(polygamma(1, x))
        outs.append(polygamma(2, x))
        results[f"test_random_{mode}"] = outs

    return results

test_results = test_polygamma()
