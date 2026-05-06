import torch
import torch.nn.functional as F

def conv2d_w8a8(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None, 
                stride: int = 1, padding: int = 0) -> torch.Tensor:
    """W8A8 conv2d benchmark (dynamic/online quantization).

    This is DYNAMIC quantization: your kernel receives fp32 inputs and must:
    1. Compute quantization scales at runtime
    2. Quantize activations and weights to int8
    3. Perform int32 accumulation during convolution
    4. Dequantize back to fp32 output

    Quantization scheme to implement:
    - Activations: per-tensor or per-channel (C_in) symmetric quantization.
    - Weights: per-output-channel (C_out) symmetric quantization (common in inference).
    - Scales computed at runtime, folded into int32 accumulation before dequantization.
    - Optional: group-wise quantization along K = C_in * K_h * K_w.

    Bias handling: bias is fp32 and added after dequantization.

    Accuracy requirement (against fp32 conv2d): cosine_sim >= 0.95, l1_relative <= 0.05, rmse <= 0.1.
    """
    return F.conv2d(input, weight, bias, stride=stride, padding=padding)

##################################################################################################################################################

import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
import torch
from data_utils import rand_tensor

def test_conv2d_w8a8():
    results = {}
    dtype = torch.float32

    # Corner case 1: 1x1 conv (pointwise; K=C_in*1*1; tests per-channel quantization)
    input_1x1 = torch.randn(1, 8, 4, 4, device='cuda', dtype=dtype)
    weight_1x1 = torch.randn(16, 8, 1, 1, device='cuda', dtype=dtype)
    results["corner_1x1_conv"] = conv2d_w8a8(input_1x1, weight_1x1, None, stride=1, padding=0)

    # Corner case 2: stride=2 and padding=0 (output size shrinks significantly)
    input_s2 = torch.randn(1, 3, 8, 8, device='cuda', dtype=dtype)
    weight_s2 = torch.randn(4, 3, 3, 3, device='cuda', dtype=dtype)
    results["corner_stride2"] = conv2d_w8a8(input_s2, weight_s2, None, stride=2, padding=0)

    # Corner case 3: all-zero kernel (one output channel kernel is all zeros)
    weight_zero = torch.randn(2, 1, 3, 3, device='cuda', dtype=dtype)
    weight_zero[0, :, :, :] = 0.0  # output channel 0 is all zeros
    input_z = torch.randn(1, 1, 5, 5, device='cuda', dtype=dtype)
    results["corner_zero_kernel"] = conv2d_w8a8(input_z, weight_zero, None, stride=1, padding=1)

    # Corner case 4: extreme bias (bias dominates conv output; tests bias add accuracy)
    input_b = torch.randn(1, 2, 4, 4, device='cuda', dtype=dtype) * 0.01  # small inputs
    weight_b = torch.randn(3, 2, 2, 2, device='cuda', dtype=dtype) * 0.01  # small weights
    bias_large = torch.tensor([100.0, -100.0, 50.0], device='cuda', dtype=dtype)  # large bias
    results["corner_large_bias"] = conv2d_w8a8(input_b, weight_b, bias_large, stride=1, padding=0)

    for mode in ("standard", "outlier"):
        outs = []
        x = rand_tensor((2, 3, 16, 16), dtype=dtype, mode=mode, outlier_prob=0.01, outlier_scale=10.0).clamp(-10, 10)
        w = rand_tensor((8, 3, 3, 3), dtype=dtype, mode=mode, outlier_prob=0.01, outlier_scale=10.0).clamp(-10, 10)
        b = rand_tensor((8,), dtype=dtype, mode=mode, outlier_prob=0.01, outlier_scale=10.0).clamp(-10, 10)
        outs.append(conv2d_w8a8(x, w, b, stride=1, padding=1))
        x2 = rand_tensor((1, 16, 8, 8), dtype=dtype, mode=mode, outlier_prob=0.01, outlier_scale=10.0).clamp(-10, 10)
        w2 = rand_tensor((16, 16, 1, 1), dtype=dtype, mode=mode, outlier_prob=0.01, outlier_scale=10.0).clamp(-10, 10)
        outs.append(conv2d_w8a8(x2, w2, None, stride=1, padding=0))
        results[f"test_random_{mode}"] = outs
    
    return results

test_results = test_conv2d_w8a8()
