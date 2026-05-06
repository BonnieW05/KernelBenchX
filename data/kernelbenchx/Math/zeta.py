import torch

def zeta(input: torch.Tensor, other: torch.Tensor, out: torch.Tensor=None) -> torch.Tensor:
    """
    Computes the Hurwitz zeta function, elementwise, for given input tensors.

    Args:
        input (torch.Tensor): the input tensor corresponding to `x`.
        other (torch.Tensor): the input tensor corresponding to `q`.
        out (torch.Tensor, optional): the output tensor. Default is None.

    Returns:
        torch.Tensor: The result of the Hurwitz zeta function computation.
    """
    return torch.special.zeta(input, other, out=out)

##################################################################################################################################################


import torch
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_tensor

def test_zeta():
    results = {}

    # Test case 1: Basic test with simple values
    input1 = torch.tensor([2.0, 3.0], device='cuda')
    other1 = torch.tensor([1.0, 2.0], device='cuda')
    results["test_case_1"] = zeta(input1, other1)

    # Test case 2: Test with larger values
    input2 = torch.tensor([10.0, 20.0], device='cuda')
    other2 = torch.tensor([5.0, 10.0], device='cuda')
    results["test_case_2"] = zeta(input2, other2)

    # Test case 3: Test with fractional values
    input3 = torch.tensor([2.5, 3.5], device='cuda')
    other3 = torch.tensor([1.5, 2.5], device='cuda')
    results["test_case_3"] = zeta(input3, other3)

    # Test case 4: Test with negative values
    input4 = torch.tensor([-2.0, -3.0], device='cuda')
    other4 = torch.tensor([1.0, 2.0], device='cuda')
    results["test_case_4"] = zeta(input4, other4)

    for mode in ("standard", "outlier"):
        x = rand_tensor((64, 64), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
        q = rand_tensor((64, 64), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
        x = x.abs() + 2.0
        q = q.abs() + 1.0
        results[f"test_random_{mode}"] = zeta(x, q)

    return results

test_results = test_zeta()
