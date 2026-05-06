import torch

def kl_div(input, target, reduction='batchmean', log_target=False):
    """
    Kullback-Leibler divergence loss.

    Args:
        input (Tensor): Log-probabilities.
        target (Tensor): Probabilities or log-probabilities.
        reduction (str): 'none' | 'batchmean' | 'sum' | 'mean'
        log_target (bool): Whether target is in log-space.

    Returns:
        Tensor: KL divergence.
    """
    return torch.nn.functional.kl_div(input, target, reduction=reduction, log_target=log_target)

##################################################################################################################################################


import torch
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_tensor

def test_kl_div():
    results = {}

    # Test case 1: input=log_softmax, target=softmax (typical distillation)
    x1 = torch.randn(4, 8, device='cuda')
    input1 = torch.log_softmax(x1, dim=-1)
    target1 = torch.softmax(torch.randn(4, 8, device='cuda'), dim=-1)
    results["test_case_1"] = kl_div(input1, target1, reduction='batchmean')

    # Test case 2: log_target=True
    x2 = torch.randn(2, 5, device='cuda')
    input2 = torch.log_softmax(x2, dim=-1)
    target2 = torch.log_softmax(torch.randn(2, 5, device='cuda'), dim=-1)
    results["test_case_2"] = kl_div(input2, target2, reduction='batchmean', log_target=True)

    # Test case 3: reduction='none'
    x3 = torch.randn(2, 3, device='cuda')
    input3 = torch.log_softmax(x3, dim=-1)
    target3 = torch.softmax(torch.randn(2, 3, device='cuda'), dim=-1)
    results["test_case_3"] = kl_div(input3, target3, reduction='none')

    for mode in ("standard", "outlier"):
        outs = []
        x = rand_tensor((16, 32), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
        inp = torch.log_softmax(x, dim=-1)
        tgt = torch.softmax(rand_tensor((16, 32), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0), dim=-1)
        outs.append(kl_div(inp, tgt, reduction='batchmean'))
        outs.append(kl_div(inp, tgt, reduction='none'))
        tgt_log = torch.log_softmax(rand_tensor((16, 32), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0), dim=-1)
        outs.append(kl_div(inp, tgt_log, reduction='batchmean', log_target=True))
        results[f"test_random_{mode}"] = outs

    return results


test_results = test_kl_div()
