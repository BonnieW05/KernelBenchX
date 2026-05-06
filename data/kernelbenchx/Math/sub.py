import torch

def sub(input: torch.Tensor, other: torch.Tensor, *, alpha: float=1, out: torch.Tensor=None) -> torch.Tensor:
    """
    Subtracts the tensor 'other' scaled by 'alpha' from the tensor 'input'.
    
    Args:
        input (torch.Tensor): The input tensor.
        other (torch.Tensor or Number): The tensor or number to subtract from input.
        alpha (float, optional): The multiplier for 'other'. Default is 1.
        out (torch.Tensor, optional): The output tensor to store the result. Default is None.
        
    Returns:
        torch.Tensor: The result of the operation, i.e., input - alpha * other.
    """
    if out is None:
        out = input - alpha * other
    else:
        torch.subtract(input, alpha * other, out=out)
    return out

##################################################################################################################################################


import torch
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_tensor

def test_sub():
    results = {}

    # Test case 1: Basic subtraction with default alpha
    input1 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    other1 = torch.tensor([0.5, 1.0, 1.5], device='cuda')
    results["test_case_1"] = sub(input1, other1)

    # Test case 2: Subtraction with alpha
    input2 = torch.tensor([4.0, 5.0, 6.0], device='cuda')
    other2 = torch.tensor([1.0, 1.0, 1.0], device='cuda')
    results["test_case_2"] = sub(input2, other2, alpha=2)

    # Test case 3: Subtraction with a scalar other
    input3 = torch.tensor([7.0, 8.0, 9.0], device='cuda')
    other3 = 2.0
    results["test_case_3"] = sub(input3, other3)

    # Test case 4: Subtraction with out parameter
    input4 = torch.tensor([10.0, 11.0, 12.0], device='cuda')
    other4 = torch.tensor([3.0, 3.0, 3.0], device='cuda')
    out4 = torch.empty(3, device='cuda')
    results["test_case_4"] = sub(input4, other4, out=out4)

    for mode in ("standard", "outlier"):
        x = rand_tensor((64, 64), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
        y = rand_tensor((64, 64), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
        results[f"test_random_{mode}"] = sub(x, y, alpha=1.0)

    return results

test_results = test_sub()
