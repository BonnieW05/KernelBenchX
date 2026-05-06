import torch

def index_select(input: torch.Tensor, dim: int, index: torch.Tensor) -> torch.Tensor:
    """Selects elements along a given dimension using a 1D index tensor.

    Args:
        input (torch.Tensor): The input tensor.
        dim (int): The dimension along which to index.
        index (torch.Tensor): 1D index tensor.

    Returns:
        torch.Tensor: The indexed tensor.
    """
    return torch.index_select(input, dim, index)

##################################################################################################################################################


import torch
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_int, rand_tensor

def test_index_select():
    results = {}

    # Test case 1: Select rows from 2D tensor
    x1 = torch.randn(6, 4, device='cuda')
    idx1 = torch.tensor([0, 2, 5], device='cuda', dtype=torch.long)
    results["test_case_1"] = index_select(x1, 0, idx1)

    # Test case 2: Select columns from 2D tensor
    x2 = torch.randn(3, 8, device='cuda')
    idx2 = torch.tensor([1, 3, 7], device='cuda', dtype=torch.long)
    results["test_case_2"] = index_select(x2, 1, idx2)

    # Test case 3: Select along last dim of 3D tensor
    x3 = torch.randn(2, 3, 10, device='cuda')
    idx3 = torch.tensor([0, 4, 9], device='cuda', dtype=torch.long)
    results["test_case_3"] = index_select(x3, 2, idx3)

    for mode in ("standard", "outlier"):
        outs = []
        for _ in range(3):
            x = rand_tensor((8, 32, 256), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=20.0)
            idx = rand_int((64,), low=0, high=256, device="cuda", dtype=torch.int64)
            outs.append(index_select(x, 2, idx))
        results[f"test_random_{mode}"] = outs

    return results


test_results = test_index_select()
