import torch

def smooth_l1_loss(input, target, beta=1.0, reduction='mean'):
    """
    Smooth L1 loss (Huber-like).

    Args:
        input (Tensor): Predicted values.
        target (Tensor): Ground truth values.
        beta (float): Transition point from L2 to L1.
        reduction (str): 'none' | 'mean' | 'sum'

    Returns:
        Tensor: Loss.
    """
    return torch.nn.functional.smooth_l1_loss(input, target, beta=beta, reduction=reduction)

##################################################################################################################################################


import torch
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_tensor

def test_smooth_l1_loss():
    results = {}

    # Test case 1: Basic usage
    input1 = torch.randn(32, device='cuda')
    target1 = torch.randn(32, device='cuda')
    results["test_case_1"] = smooth_l1_loss(input1, target1)

    # Test case 2: Smaller beta (more L1-like)
    input2 = torch.randn(32, device='cuda')
    target2 = torch.randn(32, device='cuda')
    results["test_case_2"] = smooth_l1_loss(input2, target2, beta=0.5)

    # Test case 3: reduction='none'
    input3 = torch.randn(4, 8, device='cuda')
    target3 = torch.randn(4, 8, device='cuda')
    results["test_case_3"] = smooth_l1_loss(input3, target3, reduction='none')

    for mode in ("standard", "outlier"):
        outs = []
        x1 = rand_tensor((4096,), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
        y1 = rand_tensor((4096,), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
        outs.append(smooth_l1_loss(x1, y1, beta=1.0, reduction='mean'))
        outs.append(smooth_l1_loss(x1, y1, beta=0.5, reduction='sum'))
        x2 = rand_tensor((32, 128), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
        y2 = rand_tensor((32, 128), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
        outs.append(smooth_l1_loss(x2, y2, beta=0.1, reduction='none'))
        results[f"test_random_{mode}"] = outs

    return results


test_results = test_smooth_l1_loss()
