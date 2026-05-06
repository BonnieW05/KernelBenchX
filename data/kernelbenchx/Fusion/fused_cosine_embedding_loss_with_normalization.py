import torch
import torch.nn.functional as F
import torch

def fused_cosine_embedding_loss_with_normalization(input1: torch.Tensor, input2: torch.Tensor, target: torch.Tensor, margin: float=0, reduction: str='mean') -> torch.Tensor:
    """
    Computes cosine embedding loss between two normalized tensors.
    This function first normalizes the inputs using L2 normalization and then calculates the cosine embedding loss.

    Args:
        input1 (Tensor): First input tensor to be normalized and compared.
        input2 (Tensor): Second input tensor to be normalized and compared.
        target (Tensor): Tensor label with values 1 or -1, where 1 encourages similarity and -1 encourages dissimilarity.
        margin (float, optional): Margin for dissimilarity. Default: 0.
        reduction (str, optional): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. Default: 'mean'.
    
    Returns:
        Tensor: Computed loss value.

    Example:
        input1 = torch.randn(3, 5, requires_grad=True)
        input2 = torch.randn(3, 5, requires_grad=True)
        target = torch.tensor([1, -1, 1])  # Example labels for similarity/dissimilarity
        loss = fused_cosine_embedding_loss_with_normalization(input1, input2, target)
        print(loss)
        loss.backward()
    """
    input1_normalized = F.normalize(input1, p=2, dim=1)
    input2_normalized = F.normalize(input2, p=2, dim=1)
    cosine_similarity = torch.sum(input1_normalized * input2_normalized, dim=1)
    loss = 1 - cosine_similarity * target.float()
    loss = torch.clamp(loss, min=0)
    if margin > 0:
        loss = torch.max(loss, margin - cosine_similarity)
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise ValueError(f'Invalid reduction method: {reduction}')

##################################################################################################################################################


import torch
import torch.nn.functional as F
import torch
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_tensor, rand_int

def test_fused_cosine_embedding_loss_with_normalization():
    results = {}

    # Test case 1: Default margin and reduction
    input1 = torch.randn(3, 5, device='cuda', requires_grad=True)
    input2 = torch.randn(3, 5, device='cuda', requires_grad=True)
    target = torch.tensor([1, -1, 1], device='cuda')
    results["test_case_1"] = fused_cosine_embedding_loss_with_normalization(input1, input2, target)

    # Test case 2: Margin > 0
    margin = 0.5
    results["test_case_2"] = fused_cosine_embedding_loss_with_normalization(input1, input2, target, margin=margin)

    # Test case 3: Reduction 'sum'
    reduction = 'sum'
    results["test_case_3"] = fused_cosine_embedding_loss_with_normalization(input1, input2, target, reduction=reduction)

    # Test case 4: Reduction 'none'
    reduction = 'none'
    results["test_case_4"] = fused_cosine_embedding_loss_with_normalization(input1, input2, target, reduction=reduction)

    for mode in ("standard", "outlier"):
        outs = []
        for _ in range(3):
            bsz, dim = 128, 256
            x1 = rand_tensor((bsz, dim), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            x2 = rand_tensor((bsz, dim), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            t = rand_int((bsz,), low=0, high=2, dtype=torch.int64)
            t = t * 2 - 1
            outs.append(fused_cosine_embedding_loss_with_normalization(x1, x2, t, margin=0.5, reduction="mean"))
        results[f"test_random_{mode}"] = outs

    return results

test_results = test_fused_cosine_embedding_loss_with_normalization()
