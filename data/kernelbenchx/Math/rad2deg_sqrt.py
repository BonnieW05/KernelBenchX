import torch
from typing import Tuple

def rad2deg_sqrt(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert angles from radians to degrees and calculate the square root for each element in the tensor.

    Args:
        input (Tensor): The input tensor with angles in radians.

    Returns:
        Tuple[Tensor, Tensor]: A tuple containing:
            - The first tensor with angles in degrees.
            - The second tensor with the square roots of the input tensor elements.
    
    Example:
        >>> a = torch.tensor([3.142, 1.570, 0.785, 0.0])
        >>> deg_result, sqrt_result = rad2deg_sqrt(a)
        >>> deg_result
        tensor([180.0233,  89.9544,  45.0000,   0.0000])
        >>> sqrt_result
        tensor([1.7725, 1.2533, 0.8862, 0.0000])
    """
    deg_result = torch.rad2deg(input)
    sqrt_result = torch.sqrt(input)
    return (deg_result, sqrt_result)

##################################################################################################################################################


import torch
from typing import Tuple
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_tensor

# def rad2deg_sqrt(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#     deg_result = torch.rad2deg(input)
#     sqrt_result = torch.sqrt(input)
#     return (deg_result, sqrt_result)

def test_rad2deg_sqrt():
    results = {}

    # Test case 1: Basic test with positive radians
    a = torch.tensor([3.142, 1.570, 0.785, 0.0], device='cuda')
    deg_result, sqrt_result = rad2deg_sqrt(a)
    results["test_case_1"] = (deg_result.cpu(), sqrt_result.cpu())

    # Test case 2: Test with zero
    b = torch.tensor([0.0], device='cuda')
    deg_result, sqrt_result = rad2deg_sqrt(b)
    results["test_case_2"] = (deg_result.cpu(), sqrt_result.cpu())

    # Test case 3: Test with negative radians
    c = torch.tensor([-3.142, -1.570, -0.785], device='cuda')
    deg_result, sqrt_result = rad2deg_sqrt(c)
    results["test_case_3"] = (deg_result.cpu(), sqrt_result.cpu())

    # Test case 4: Test with a mix of positive and negative radians
    d = torch.tensor([3.142, -1.570, 0.785, -0.785], device='cuda')
    deg_result, sqrt_result = rad2deg_sqrt(d)
    results["test_case_4"] = (deg_result.cpu(), sqrt_result.cpu())

    for mode in ("standard", "outlier"):
        outs = []
        x = rand_tensor((64, 64), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
        x = x.abs()
        outs.append(rad2deg_sqrt(x))
        outs.append(rad2deg_sqrt(x + 1e-3))
        results[f"test_random_{mode}"] = outs

    return results

test_results = test_rad2deg_sqrt()
