import torch

def nll_loss(input, target, weight=None, ignore_index=-100, reduction='mean'):
    """
    Negative Log Likelihood loss.
    
    Args:
        input (Tensor): Log probabilities of shape (N, C) or (N, C, d1, ..., dk)
        target (Tensor): Ground truth class indices
        weight (Tensor, optional): Manual rescaling weight for each class
        ignore_index (int): Specifies a target value that is ignored
        reduction (str): 'none' | 'mean' | 'sum'
        
    Returns:
        Tensor: Computed NLL loss
    """
    return torch.nn.functional.nll_loss(input, target, weight=weight, 
                                        ignore_index=ignore_index, reduction=reduction)

##################################################################################################################################################


import torch
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_tensor

def test_nll_loss():
    results = {}

    # Test case 1: Basic usage with log probabilities
    input1 = torch.randn(4, 3, device='cuda').log_softmax(dim=1)
    target1 = torch.tensor([0, 1, 2, 1], device='cuda')
    results["test_case_1"] = nll_loss(input1, target1)

    # Test case 2: With class weights
    input2 = torch.randn(4, 3, device='cuda').log_softmax(dim=1)
    target2 = torch.tensor([0, 1, 2, 0], device='cuda')
    weight2 = torch.tensor([1.0, 2.0, 1.5], device='cuda')
    results["test_case_2"] = nll_loss(input2, target2, weight=weight2)

    # Test case 3: With ignore_index
    input3 = torch.randn(4, 3, device='cuda').log_softmax(dim=1)
    target3 = torch.tensor([0, -100, 2, 1], device='cuda')
    results["test_case_3"] = nll_loss(input3, target3, ignore_index=-100)

    for mode in ("standard", "outlier"):
        outs = []
        C = 10
        logits = rand_tensor((32, C), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
        logp = torch.log_softmax(logits, dim=1)
        target = torch.randint(0, C, (32,), device='cuda')
        outs.append(nll_loss(logp, target))
        weight = torch.rand((C,), device='cuda', dtype=torch.float32) + 0.1
        outs.append(nll_loss(logp, target, weight=weight))
        target_ign = target.clone()
        target_ign[0] = -100
        outs.append(nll_loss(logp, target_ign, ignore_index=-100))
        results[f"test_random_{mode}"] = outs

    return results

test_results = test_nll_loss()
