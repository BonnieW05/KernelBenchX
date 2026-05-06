import torch

def cross_entropy(input, target, weight=None, ignore_index=-100, reduction='mean'):
    """
    Cross entropy loss between input logits and target labels.
    
    Args:
        input (Tensor): Predicted logits of shape (N, C) or (N, C, d1, ..., dk)
        target (Tensor): Ground truth class indices of shape (N,) or (N, d1, ..., dk)
        weight (Tensor, optional): Manual rescaling weight for each class
        ignore_index (int): Specifies a target value that is ignored
        reduction (str): 'none' | 'mean' | 'sum'
        
    Returns:
        Tensor: Computed loss
    """
    return torch.nn.functional.cross_entropy(input, target, weight=weight, 
                                             ignore_index=ignore_index, reduction=reduction)

##################################################################################################################################################


import torch
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_tensor

def test_cross_entropy():
    results = {}

    # Test case 1: Basic 2D classification (batch_size=4, num_classes=3)
    input1 = torch.randn(4, 3, device='cuda', requires_grad=True)
    target1 = torch.tensor([0, 1, 2, 1], device='cuda')
    results["test_case_1"] = cross_entropy(input1, target1)

    # Test case 2: With ignore_index (useful for padding tokens)
    input2 = torch.randn(4, 3, device='cuda')
    target2 = torch.tensor([0, -100, 2, 1], device='cuda')  # -100 will be ignored
    results["test_case_2"] = cross_entropy(input2, target2, ignore_index=-100)

    # Test case 3: 3D input for image segmentation (N, C, H, W)
    input3 = torch.randn(2, 3, 4, 4, device='cuda')
    target3 = torch.randint(0, 3, (2, 4, 4), device='cuda')
    results["test_case_3"] = cross_entropy(input3, target3)

    for mode in ("standard", "outlier"):
        outs = []
        C = 10
        logits = rand_tensor((32, C), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
        target = torch.randint(0, C, (32,), device='cuda')
        outs.append(cross_entropy(logits, target))
        target_ign = target.clone()
        target_ign[0] = -100
        outs.append(cross_entropy(logits, target_ign, ignore_index=-100))
        logits2 = rand_tensor((2, C, 8, 8), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
        target2 = torch.randint(0, C, (2, 8, 8), device='cuda')
        outs.append(cross_entropy(logits2, target2))
        results[f"test_random_{mode}"] = outs

    return results

test_results = test_cross_entropy()
