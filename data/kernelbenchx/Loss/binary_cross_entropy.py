import torch

def binary_cross_entropy(input, target, weight=None, reduction='mean'):
    """
    Binary cross entropy loss.

    Args:
        input (Tensor): Probabilities in [0, 1], same shape as target.
        target (Tensor): Targets in {0, 1} or [0, 1], same shape as input.
        weight (Tensor, optional): A manual rescaling weight given to each loss element.
        reduction (str): 'none' | 'mean' | 'sum'

    Returns:
        Tensor: Loss.
    """
    return torch.nn.functional.binary_cross_entropy(input, target, weight=weight, reduction=reduction)

##################################################################################################################################################


import torch
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_tensor

def test_binary_cross_entropy():
    results = {}

    # Test case 1: Basic BCE on probabilities
    logits1 = torch.randn(8, device='cuda')
    input1 = torch.sigmoid(logits1)
    target1 = torch.randint(0, 2, (8,), device='cuda').float()
    results["test_case_1"] = binary_cross_entropy(input1, target1)

    # Test case 2: With element-wise weight
    logits2 = torch.randn(8, device='cuda')
    input2 = torch.sigmoid(logits2)
    target2 = torch.randint(0, 2, (8,), device='cuda').float()
    weight2 = torch.rand(8, device='cuda')
    results["test_case_2"] = binary_cross_entropy(input2, target2, weight=weight2)

    # Test case 3: reduction='none' (per-element loss)
    logits3 = torch.randn(4, 4, device='cuda')
    input3 = torch.sigmoid(logits3)
    target3 = torch.randint(0, 2, (4, 4), device='cuda').float()
    results["test_case_3"] = binary_cross_entropy(input3, target3, reduction='none')

    for mode in ("standard", "outlier"):
        outs = []
        for shape in ((1024,), (64, 128)):
            logits = rand_tensor(shape, dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0).clamp(-20, 20)
            inputp = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
            target = torch.sigmoid(rand_tensor(shape, dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0))
            outs.append(binary_cross_entropy(inputp, target, reduction='mean'))
            weight = torch.rand(shape, device='cuda', dtype=torch.float32)
            outs.append(binary_cross_entropy(inputp, target, weight=weight, reduction='mean'))
            outs.append(binary_cross_entropy(inputp, target, reduction='none'))
        results[f"test_random_{mode}"] = outs

    return results


test_results = test_binary_cross_entropy()
