import torch

def masked_select(input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Selects elements of the input tensor according to a boolean mask.

    Args:
        input (torch.Tensor): The input tensor.
        mask (torch.Tensor): A boolean mask tensor broadcastable to input.

    Returns:
        torch.Tensor: A 1D tensor containing the selected elements.
    """
    return torch.masked_select(input, mask)

##################################################################################################################################################


import torch
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_bool, rand_tensor

def test_masked_select():
    results = {}

    # Test case 1: 1D masked select
    x1 = torch.randn(16, device='cuda')
    m1 = (x1 > 0)
    results["test_case_1"] = masked_select(x1, m1)

    # Test case 2: 2D masked select
    x2 = torch.randn(4, 5, device='cuda')
    m2 = (torch.rand(4, 5, device='cuda') > 0.5)
    results["test_case_2"] = masked_select(x2, m2)

    # Test case 3: Broadcastable mask
    x3 = torch.randn(2, 3, 4, device='cuda')
    m3 = (torch.rand(1, 3, 1, device='cuda') > 0.3)
    results["test_case_3"] = masked_select(x3, m3)

    for mode in ("standard", "outlier"):
        outs = []
        for _ in range(3):
            x = rand_tensor((64, 128), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=20.0)
            m = rand_bool((64, 128))
            outs.append(masked_select(x, m))
        results[f"test_random_{mode}"] = outs

    return results


test_results = test_masked_select()
