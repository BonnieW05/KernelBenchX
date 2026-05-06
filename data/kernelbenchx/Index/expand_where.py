import torch


def expand_where(input, target_sizes, cond, other):
    """    
    Expand the input tensor via broadcasting and then select values using torch.where.

    Args:
        input (Tensor): Tensor that will be expanded (typically with singleton dimensions).
        target_sizes (tuple of int): Target sizes passed to expand().
        cond (Tensor): Boolean condition tensor broadcastable to the expanded shape.
        other (Tensor): Tensor broadcastable to the expanded shape.

    Returns:
        Tensor: Result tensor.
    """
    expanded = input.expand(*target_sizes)
    return torch.where(cond, expanded, other)


##################################################################################################################################################


import torch
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_bool, rand_tensor


def test_expand_where():
    results = {}

    x = torch.randn(1, 4096, device='cuda')
    target_sizes = (256, 4096)
    cond = (torch.arange(256, device='cuda') % 2 == 0).view(256, 1)
    y = torch.randn(256, 4096, device='cuda')

    results["test_case_1"] = expand_where(x, target_sizes, cond, y)

    x2 = torch.randn(1, 128, device='cuda')
    target_sizes2 = (32, 128)
    cond2 = torch.randint(0, 2, (32, 1), device='cuda', dtype=torch.bool)
    y2 = torch.zeros(32, 128, device='cuda')

    results["test_case_2"] = expand_where(x2, target_sizes2, cond2, y2)

    for mode in ("standard", "outlier"):
        outs = []
        for _ in range(3):
            x = rand_tensor((1, 512), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            target_sizes = (64, 512)
            cond = rand_bool((64, 1))
            other = rand_tensor((64, 512), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            outs.append(expand_where(x, target_sizes, cond, other))
        results[f"test_random_{mode}"] = outs

    return results


test_results = test_expand_where()

