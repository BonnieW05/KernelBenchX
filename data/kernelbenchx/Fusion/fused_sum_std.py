import torch

def fused_sum_std(input, dim=None, keepdim=False, dtype=None, correction=1, out=None):
    """
    Computes the sum of elements in the input tensor along the specified dimension(s),
    followed by calculating the standard deviation of the summed values.
    
    Args:
        input (Tensor): The input tensor.
        dim (int or tuple of ints, optional): The dimension(s) to reduce. If None, all dimensions are reduced.
        keepdim (bool, optional): Whether the output tensor has dim retained or not. Default is False.
        dtype (torch.dtype, optional): The desired data type of the returned tensor. Default: None.
        correction (int, optional): Difference between the sample size and sample degrees of freedom. Default is 1.
        out (Tensor, optional): The output tensor.
    
    Returns:
        Tensor: A tensor containing the standard deviation of the summed values along the specified dimension(s).
    """
    summed = input.sum(dim=dim, keepdim=keepdim, dtype=dtype)
    n = summed.numel()
    mean = summed.mean()
    var = ((summed - mean) ** 2).sum()
    if n > correction:
        std = (var / (n - correction)).sqrt()
    else:
        std = torch.tensor(0.0, dtype=summed.dtype)
    return std

##################################################################################################################################################


import torch
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_tensor

def test_sum_std():
    results = {}
    
    # Test case 1: Basic test with a 1D tensor
    input1 = torch.tensor([1.0, 2.0, 3.0, 4.0], device='cuda')
    results["test_case_1"] = fused_sum_std(input1)

    # Test case 2: Test with a 2D tensor along dim=0
    input2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    results["test_case_2"] = fused_sum_std(input2, dim=0)

    # Test case 3: Test with a 2D tensor along dim=1
    input3 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    results["test_case_3"] = fused_sum_std(input3, dim=1)

    # Test case 4: Test with keepdim=True
    input4 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    results["test_case_4"] = fused_sum_std(input4, dim=0, keepdim=True)

    for mode in ("standard", "outlier"):
        outs = []
        for dim in (None, 0, 1):
            x = rand_tensor((128, 256), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            outs.append(fused_sum_std(x, dim=dim, keepdim=False, correction=1))
        results[f"test_random_{mode}"] = outs

    return results

test_results = test_sum_std()
