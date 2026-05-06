import torch
import math
from typing import Tuple

def erfc_sqrt(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the complementary error function (erfc) and the square root of each element in the input tensor.

    Args:
        input (torch.Tensor): The input tensor for which the erfc and square root are computed.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - erfc_result (torch.Tensor): The complementary error function results.
            - sqrt_result (torch.Tensor): The square root results.
    """
    erfc_result = torch.erfc(input)
    sqrt_result = torch.sqrt(input)
    sqrt_result[input < 0] = float('nan')
    return (erfc_result, sqrt_result)

##################################################################################################################################################


import torch
import math
from typing import Tuple
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_tensor

# def erfc_sqrt(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     Computes the complementary error function (erfc) and the square root of each element in the input tensor.

#     Args:
#         input (torch.Tensor): The input tensor for which the erfc and square root are computed.

#     Returns:
#         Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
#             - erfc_result (torch.Tensor): The complementary error function results.
#             - sqrt_result (torch.Tensor): The square root results.
#     """
#     erfc_result = torch.erfc(input)
#     sqrt_result = torch.sqrt(input)
#     sqrt_result[input < 0] = float('nan')
#     return (erfc_result, sqrt_result)

def test_erfc_sqrt():
    results = {}

    # Test case 1: Positive values
    input1 = torch.tensor([0.0, 1.0, 2.0], device='cuda')
    results["test_case_1"] = erfc_sqrt(input1)

    # Test case 2: Negative values
    input2 = torch.tensor([-1.0, -2.0, -3.0], device='cuda')
    results["test_case_2"] = erfc_sqrt(input2)

    # Test case 3: Mixed values
    input3 = torch.tensor([-1.0, 0.0, 1.0], device='cuda')
    results["test_case_3"] = erfc_sqrt(input3)

    # Test case 4: Zero values
    input4 = torch.tensor([0.0, 0.0, 0.0], device='cuda')
    results["test_case_4"] = erfc_sqrt(input4)

    for mode in ("standard", "outlier"):
        x = rand_tensor((64, 64), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
        x = x.abs()
        results[f"test_random_{mode}"] = erfc_sqrt(x)

    return results

test_results = test_erfc_sqrt()
