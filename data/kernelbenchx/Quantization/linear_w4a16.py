import torch
import torch.nn.functional as F

def linear_w4a16(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
    """W4A16 linear benchmark (weight-only 4-bit dynamic quantization).

    Quantization scheme to implement:
    - Input (activations): fp16, NOT quantized, used as-is
    - Weight: symmetric int4, quantize at runtime inside kernel
    * Scale: per-output-channel, scale[n] = max(abs(weight[n, :])) / 7.0
    * Clamp to [-8, 7] (int4 signed range), pack two values per byte
    * Packing order convention: low nibble = first element (standard convention)
    - Dequantize weights on-the-fly: w_fp16 = int4_val * scale[n]
    - Accumulate in fp16 or fp32, output in fp16
    - Optional: group-wise quantization (group_size=64 along K) for better accuracy

    Accuracy requirement (against fp16 linear): cosine_sim >= 0.90, l1_relative <= 0.10, rmse <= 0.15.
    """
    return F.linear(input, weight, bias)

##################################################################################################################################################

import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
import torch
from data_utils import rand_tensor

def test_linear_w4a16():
    results = {}
    dtype = torch.float16

    # Corner case 1: int4 boundary values (weights quantize exactly to -8 or 7)
    weight_boundary = torch.tensor(
        [[7.0, -8.0, 3.5, -3.5]],  # designed to hit int4 boundaries after quantization
        device='cuda', dtype=dtype
    )
    input_bd = torch.tensor([[1.0, 1.0, 1.0, 1.0]], device='cuda', dtype=dtype)
    results["corner_int4_boundary"] = linear_w4a16(input_bd, weight_boundary, None)

    # Corner case 2: extreme scale differences (tests per-output-channel scales)
    weight_extreme = torch.tensor(
        [[100.0, 0.1, 0.1, 0.1],   # row 1 has a huge max -> large scale, other values lose precision
         [0.1, 0.1, 0.1, 0.1]],    # row 2 has a normal scale
        device='cuda', dtype=dtype
    )
    input_ex = torch.ones(1, 4, device='cuda', dtype=dtype)
    results["corner_extreme_scale"] = linear_w4a16(input_ex, weight_extreme, None)

    # Corner case 3: all-zero row (one output neuron's weights are all zeros)
    weight_zero_row = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0],
         [0.0, 0.0, 0.0, 0.0],  # all-zero row -> scale=0 edge case
         [5.0, 6.0, 7.0, 8.0]],
        device='cuda', dtype=dtype
    )
    input_zr = torch.randn(2, 4, device='cuda', dtype=dtype)
    results["corner_zero_row"] = linear_w4a16(input_zr, weight_zero_row, None)

    # Corner case 4: single output dimension (Dx1 linear layer)
    weight_1d = torch.tensor([[0.5, -0.5, 1.0, -1.0]], device='cuda', dtype=dtype)
    input_1d = torch.tensor([[2.0, 2.0, 2.0, 2.0]], device='cuda', dtype=dtype)
    bias_1d = torch.tensor([10.0], device='cuda', dtype=dtype)
    results["corner_single_output"] = linear_w4a16(input_1d, weight_1d, bias_1d)

    for mode in ("standard", "outlier"):
        outs = []
        for B, D_in, D_out in ((16, 256, 128), (8, 128, 64)):
            x = rand_tensor((B, D_in), dtype=dtype, mode=mode, outlier_prob=0.01, outlier_scale=10.0).clamp(-10, 10)
            w = rand_tensor((D_out, D_in), dtype=dtype, mode=mode, outlier_prob=0.01, outlier_scale=10.0).clamp(-10, 10)
            outs.append(linear_w4a16(x, w, None))
        results[f"test_random_{mode}"] = outs
    
    return results

test_results = test_linear_w4a16()
