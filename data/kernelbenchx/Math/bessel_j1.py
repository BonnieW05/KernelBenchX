import torch


def bessel_j1(input: torch.Tensor, out: torch.Tensor=None) -> torch.Tensor:
    """
    Computes the Bessel function of the first kind of order 1 for each element of the input tensor.

    Args:
        input (torch.Tensor): The input tensor.
        out (torch.Tensor, optional): The output tensor. If provided, the result will be stored in this tensor.

    Returns:
        torch.Tensor: The result of applying the Bessel function of the first kind of order 1 to each element in the input tensor.
    """
    return torch.special.bessel_j1(input, out=out)

##################################################################################################################################################


import torch
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_tensor

def test_bessel_j1():
    results = {}

    # Test case 1: Basic test with a single positive value
    input1 = torch.tensor([1.0], device='cuda')
    results["test_case_1"] = bessel_j1(input1)

    # Test case 2: Test with a tensor of multiple values
    input2 = torch.tensor([0.0, 1.0, 2.0, 3.0], device='cuda')
    results["test_case_2"] = bessel_j1(input2)

    # Test case 3: Test with a tensor of negative values
    input3 = torch.tensor([-1.0, -2.0, -3.0], device='cuda')
    results["test_case_3"] = bessel_j1(input3)

    # Test case 4: Test with a larger tensor
    input4 = torch.linspace(-5.0, 5.0, steps=10, device='cuda')
    results["test_case_4"] = bessel_j1(input4)

    for mode in ("standard", "outlier"):
        outs = []
        for shape in ((1024,), (64, 64)):
            x = rand_tensor(shape, dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            outs.append(bessel_j1(x))
        results[f"test_random_{mode}"] = outs

    return results

test_results = test_bessel_j1()
