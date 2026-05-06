import torch

def index_fill_(dim, x, index, value):
    """
    Fill the tensor `x` at the positions specified by `index` along dimension `dim`
    with the given `value`.
    
    Args:
    - dim (int): The dimension along which to index.
    - x (torch.Tensor): The input tensor.
    - index (torch.Tensor): A tensor containing the indices.
    - value (int or float): The value to fill at the indexed positions.
    
    Returns:
    - torch.Tensor: The updated tensor.
    """
    return x.index_fill_(dim, index, value)

##################################################################################################################################################


import torch
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_int, rand_tensor

def test_index_fill_():
    results = {}

    # Test case 1: Basic functionality
    x1 = torch.zeros((3, 3), device='cuda')
    index1 = torch.tensor([0, 2], device='cuda')
    value1 = 5
    results["test_case_1"] = index_fill_(0, x1, index1, value1).cpu()

    # Test case 2: Different dimension
    x2 = torch.zeros((3, 3), device='cuda')
    index2 = torch.tensor([1], device='cuda')
    value2 = 3
    results["test_case_2"] = index_fill_(1, x2, index2, value2).cpu()

    # Test case 3: Single element tensor
    x3 = torch.zeros((1, 1), device='cuda')
    index3 = torch.tensor([0], device='cuda')
    value3 = 7
    results["test_case_3"] = index_fill_(0, x3, index3, value3).cpu()

    # Test case 4: Larger tensor
    x4 = torch.zeros((5, 5), device='cuda')
    index4 = torch.tensor([1, 3, 4], device='cuda')
    value4 = 9
    results["test_case_4"] = index_fill_(0, x4, index4, value4).cpu()

    for mode in ("standard", "outlier"):
        outs = []
        for _ in range(3):
            x = rand_tensor((64, 128), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=20.0)
            idx = rand_int((16,), low=0, high=64, device="cuda", dtype=torch.int64)
            outs.append(index_fill_(0, x, idx, 3.14).cpu())
        results[f"test_random_{mode}"] = outs

    return results

test_results = test_index_fill_()
