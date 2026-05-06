import torch
import torch.nn.functional as F


def attention(q, k, v, causal: bool = False, softmax_scale=None, *, out=None):
    """Compute scaled dot-product attention.

    Args:
        q (Tensor): Query tensor of shape (B, H, S, D).
        k (Tensor): Key tensor of shape (B, H, S, D).
        v (Tensor): Value tensor of shape (B, H, S, D).
        causal (bool, optional): If True, apply a causal mask. Default: False.
        softmax_scale (float, optional): Scale factor applied to attention logits.
            If None, uses 1/sqrt(D).
        out (Tensor, optional): Output tensor.

    Returns:
        Tensor: Attention output of shape (B, H, S, D).
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * softmax_scale

    if causal:
        s = q.shape[2]
        mask = torch.triu(torch.ones(s, s, device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float('-inf'))

    attn = F.softmax(scores, dim=-1)
    output = torch.matmul(attn, v.float())

    if out is not None:
        out.copy_(output.to(out.dtype))
        return out
    return output.to(q.dtype)


##################################################################################################################################################


import torch
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_tensor


def test_attention():
    results = {}

    b, h, s, d = 2, 4, 128, 64
    q = torch.randn(b, h, s, d, device='cuda', dtype=torch.float16)
    k = torch.randn(b, h, s, d, device='cuda', dtype=torch.float16)
    v = torch.randn(b, h, s, d, device='cuda', dtype=torch.float16)
    results["test_case_1"] = attention(q, k, v)

    b, h, s, d = 2, 4, 127, 64
    q = torch.randn(b, h, s, d, device='cuda', dtype=torch.float16)
    k = torch.randn(b, h, s, d, device='cuda', dtype=torch.float16)
    v = torch.randn(b, h, s, d, device='cuda', dtype=torch.float16)
    results["test_case_2"] = attention(q, k, v, causal=True)

    b, h, s, d = 1, 2, 64, 32
    q = torch.randn(b, h, s, d, device='cuda', dtype=torch.float32)
    k = torch.randn(b, h, s, d, device='cuda', dtype=torch.float32)
    v = torch.randn(b, h, s, d, device='cuda', dtype=torch.float32)
    out = torch.empty_like(q)
    results["test_case_3"] = attention(q, k, v, softmax_scale=0.125, out=out)

    for mode in ("standard", "outlier"):
        for causal in (False, True):
            outs = []
            for _ in range(2):
                b, h, s, d = 2, 4, 64, 32
                q = rand_tensor((b, h, s, d), dtype=torch.float16, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
                k = rand_tensor((b, h, s, d), dtype=torch.float16, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
                v = rand_tensor((b, h, s, d), dtype=torch.float16, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
                outs.append(attention(q, k, v, causal=causal))
            results[f"test_random_{mode}_{'causal' if causal else 'noncausal'}"] = outs

    return results


test_results = test_attention()
