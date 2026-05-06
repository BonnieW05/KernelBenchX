import torch

def scatter(input, dim, index, src):
    """
    Scatter values from src into input at positions specified by index.
    
    Args:
        input (Tensor): Destination tensor
        dim (int): Dimension along which to scatter
        index (Tensor): Indices where to scatter
        src (Tensor): Source values to scatter
        
    Returns:
        Tensor: Result after scattering
    """
    return input.scatter(dim, index, src)

##################################################################################################################################################


import torch
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_int, rand_tensor

def test_scatter():
    results = {}

    # Test case 1: Scatter along dim=0
    input1 = torch.zeros(4, 3, device='cuda')
    index1 = torch.tensor([[0, 1, 2], [1, 2, 0]], device='cuda')
    src1 = torch.randn(2, 3, device='cuda')
    results["test_case_1"] = scatter(input1, 0, index1, src1)

    # Test case 2: Scatter along dim=1
    input2 = torch.zeros(3, 4, device='cuda')
    index2 = torch.tensor([[0, 2], [1, 3], [2, 0]], device='cuda')
    src2 = torch.randn(3, 2, device='cuda')
    results["test_case_2"] = scatter(input2, 1, index2, src2)

    # Test case 3: Scatter in 3D tensor
    input3 = torch.zeros(2, 3, 4, device='cuda')
    index3 = torch.tensor([[[0, 1], [1, 2], [2, 3]]], device='cuda').expand(2, 3, 2)
    src3 = torch.randn(2, 3, 2, device='cuda')
    results["test_case_3"] = scatter(input3, 2, index3, src3)

    for mode in ("standard", "outlier"):
        outs = []
        for _ in range(3):
            base = rand_tensor((4, 128, 256), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=20.0)
            idx = rand_int((4, 128, 64), low=0, high=256, device="cuda", dtype=torch.int64)
            src = rand_tensor((4, 128, 64), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=20.0)
            outs.append(scatter(base, 2, idx, src))
        results[f"test_random_{mode}"] = outs

    return results

test_results = test_scatter()
