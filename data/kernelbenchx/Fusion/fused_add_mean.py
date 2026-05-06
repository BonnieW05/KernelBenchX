import torch

def fused_add_mean(input, other, dim=None, alpha=1, keepdim=False, dtype=None, out=None):
    """
    Adds the `other` tensor, scaled by `alpha`, to the `input` tensor and computes the mean value
    along the specified dimension(s).
    
    Parameters:
        input (Tensor): The input tensor.
        other (Tensor or Number): The tensor or number to add to input.
        dim (int or tuple of ints, optional): The dimension(s) to reduce. Default: None.
        alpha (Number, optional): The multiplier for `other`. Default: 1.
        keepdim (bool, optional): Whether the output tensor has dim retained or not. Default: False.
        dtype (torch.dtype, optional): The desired data type of the returned tensor. Default: None.
        out (Tensor, optional): The output tensor.

    Returns:
        Tensor: A tensor containing the mean of the result after addition and scaling.
    """
    if isinstance(other, (int, float)):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    result = input + alpha * other
    mean_result = result.mean(dim=dim, keepdim=keepdim, dtype=dtype)
    return mean_result

##################################################################################################################################################


import torch
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_tensor

def test_add_mean():
    results = {}

    # Test case 1: Basic addition and mean with default alpha
    input1 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    other1 = torch.tensor([0.5, 0.5, 0.5], device='cuda')
    results["test_case_1"] = fused_add_mean(input1, other1)

    # Test case 2: Addition with scalar other and non-default alpha
    input2 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    other2 = 0.5
    results["test_case_2"] = fused_add_mean(input2, other2, alpha=2)

    # Test case 3: Addition with mean along a specific dimension
    input3 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    other3 = torch.tensor([[0.5, 0.5], [0.5, 0.5]], device='cuda')
    results["test_case_3"] = fused_add_mean(input3, other3, dim=0)

    # Test case 4: Addition with mean and keepdim=True
    input4 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    other4 = torch.tensor([[0.5, 0.5], [0.5, 0.5]], device='cuda')
    results["test_case_4"] = fused_add_mean(input4, other4, dim=1, keepdim=True)

    for mode in ("standard", "outlier"):
        for dim in (None, 0, 1):
            outs = []
            for _ in range(2):
                x = rand_tensor((128, 256), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
                y = rand_tensor((128, 256), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
                outs.append(fused_add_mean(x, y, dim=dim, alpha=0.5, keepdim=True))
            results[f"test_random_{mode}_dim{dim}"] = outs

    return results

test_results = test_add_mean()
