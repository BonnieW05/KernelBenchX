import torch
import torch.nn.functional as F

def fused_embedding_add_tanh(input_indices, weight, other, *, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, out=None):
    embeddings = F.embedding(input_indices, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)
    sum_embeddings = embeddings + other
    result = torch.tanh(sum_embeddings)
    if out is not None:
        out.copy_(result)
    return result

##################################################################################################################################################


import torch
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_tensor, rand_int

# def fused_embedding_add_tanh(input_indices, weight, other, *, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, out=None):
#     embeddings = F.embedding(input_indices, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)
#     sum_embeddings = embeddings + other
#     result = torch.tanh(sum_embeddings)
#     if out is not None:
#         out.copy_(result)
#     return result

def test_fused_embedding_add_tanh():
    results = {}

    # Test case 1: Basic test without padding_idx, max_norm, scale_grad_by_freq, sparse, and out
    input_indices = torch.tensor([1, 2, 3], device='cuda')
    weight = torch.randn(5, 3, device='cuda')
    other = torch.randn(3, 3, device='cuda')
    results["test_case_1"] = fused_embedding_add_tanh(input_indices, weight, other)

    # Test case 2: Test with padding_idx
    padding_idx = 0
    input_indices = torch.tensor([0, 1, 2], device='cuda')
    weight = torch.randn(5, 3, device='cuda')
    other = torch.randn(3, 3, device='cuda')
    results["test_case_2"] = fused_embedding_add_tanh(input_indices, weight, other, padding_idx=padding_idx)

    # Test case 3: Test with max_norm
    max_norm = 1.0
    input_indices = torch.tensor([1, 2, 3], device='cuda')
    weight = torch.randn(5, 3, device='cuda')
    other = torch.randn(3, 3, device='cuda')
    results["test_case_3"] = fused_embedding_add_tanh(input_indices, weight, other, max_norm=max_norm)

    # Test case 4: Test with norm_type
    norm_type = 1.0
    input_indices = torch.tensor([1, 2, 3], device='cuda')
    weight = torch.randn(5, 3, device='cuda')
    other = torch.randn(3, 3, device='cuda')
    results["test_case_4"] = fused_embedding_add_tanh(input_indices, weight, other, norm_type=norm_type)

    for mode in ("standard", "outlier"):
        outs = []
        for _ in range(2):
            bsz, seqlen, vocab, dim = 4, 16, 128, 64
            idx = rand_int((bsz, seqlen), low=0, high=vocab, dtype=torch.int64)
            w = rand_tensor((vocab, dim), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            other = rand_tensor((bsz, seqlen, dim), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            outs.append(fused_embedding_add_tanh(idx, w, other, padding_idx=0))
        results[f"test_random_{mode}"] = outs

    return results

test_results = test_fused_embedding_add_tanh()
