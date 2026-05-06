import torch
import torch.nn.functional as F

def attention_w8a8(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """W8A8 attention benchmark (dynamic/online quantization).

    This is DYNAMIC quantization: your kernel receives fp32 Q/K/V and must:
    1. Compute quantization scales at runtime for Q/K/V
    2. Quantize Q/K/V to int8
    3. Perform QK^T matmul with int32 accumulation
    4. Dequantize to fp32 before softmax
    5. Softmax in fp32 (must NOT be quantized)
    6. Attention weights @ V matmul: dequantize V to fp32 first, then fp32 @ fp32 (or advanced: keep V as int8, fp32 @ int8 with on-the-fly dequant)
    7. Output shape: same as value, i.e. (B, S, D), dtype=float32

    Quantization scheme to implement:
    - Q/K/V: per-row (token-wise) symmetric quantization along S dimension.
    - The 1/sqrt(D) scaling should be fused before quantization to reduce dynamic range.
    - Softmax must remain in fp32 (no quantization).
    - Attention probabilities quantization is optional extra credit (significantly harder).

    Accuracy requirement (against fp32 attention): cosine_sim >= 0.90, l1_relative <= 0.10, rmse <= 0.15.
    """
    scores = torch.matmul(query, key.transpose(-2, -1)) / (query.size(-1) ** 0.5)
    attn_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, value)

##################################################################################################################################################

import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
import torch
from data_utils import rand_tensor

def test_attention_w8a8():
    results = {}
    dtype = torch.float32

    # Corner case 1: single-token sequence (S=1; softmax degenerates to 1.0)
    q1 = torch.randn(1, 1, 8, device='cuda', dtype=dtype)
    k1 = torch.randn(1, 1, 8, device='cuda', dtype=dtype)
    v1 = torch.randn(1, 1, 8, device='cuda', dtype=dtype)
    results["corner_single_token"] = attention_w8a8(q1, k1, v1)

    # Corner case 2: extreme attention distribution (QK^T makes softmax near one-hot)
    q_extreme = torch.tensor([[[10.0, 0.0], [0.0, 10.0]]], device='cuda', dtype=dtype)
    k_extreme = torch.tensor([[[10.0, 0.0], [0.0, 10.0]]], device='cuda', dtype=dtype)
    v_extreme = torch.randn(1, 2, 2, device='cuda', dtype=dtype)
    results["corner_extreme_attention"] = attention_w8a8(q_extreme, k_extreme, v_extreme)

    # Corner case 3: all-zero query/key (tests softmax numerical stability)
    q_zero = torch.zeros(1, 3, 4, device='cuda', dtype=dtype)
    k_zero = torch.zeros(1, 3, 4, device='cuda', dtype=dtype)
    v_norm = torch.randn(1, 3, 4, device='cuda', dtype=dtype)
    results["corner_zero_qk"] = attention_w8a8(q_zero, k_zero, v_norm)

    # Random tests (standard + outlier)
    for mode in ("standard", "outlier"):
        outs = []
        for B, S, D in ((2, 64, 32), (1, 32, 16)):
            query = rand_tensor((B, S, D), dtype=dtype, mode=mode, outlier_prob=0.01, outlier_scale=10.0).clamp(-10, 10)
            key = rand_tensor((B, S, D), dtype=dtype, mode=mode, outlier_prob=0.01, outlier_scale=10.0).clamp(-10, 10)
            value = rand_tensor((B, S, D), dtype=dtype, mode=mode, outlier_prob=0.01, outlier_scale=10.0).clamp(-10, 10)
            outs.append(attention_w8a8(query, key, value))
        results[f"random_{mode}"] = outs
    
    return results

test_results = test_attention_w8a8()
