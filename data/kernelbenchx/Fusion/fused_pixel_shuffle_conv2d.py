import torch
import torch.nn.functional as F

def fused_pixel_shuffle_conv2d(input: torch.Tensor, weight: torch.Tensor, bias=None, stride=1, padding=0, dilation=1, groups=1, upscale_factor=2) -> torch.Tensor:
    """
    Applies a 2D convolution followed by pixel shuffle upscaling to rearrange the spatial dimensions.

    Parameters:
    - input (Tensor): Input tensor of shape (minibatch, in_channels, iH, iW).
    - weight (Tensor): Convolution filter tensor of shape (out_channels, in_channels/groups, kH, kW).
    - bias (Tensor, optional): Optional bias tensor of shape (out_channels).
    - stride (int, optional): Stride of the convolving kernel. Default is 1.
    - padding (int, optional): Padding added to all four sides of the input. Default is 0.
    - dilation (int, optional): Spacing between kernel elements. Default is 1.
    - groups (int, optional): Number of blocked connections from input channels to output channels. Default is 1.
    - upscale_factor (int, optional): Factor by which to increase spatial resolution. Default is 2.

    Returns:
    - Tensor: The output tensor after applying the convolution and pixel shuffle.
    """
    x = F.conv2d(input, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
    return F.pixel_shuffle(x, upscale_factor)

##################################################################################################################################################


import torch
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_tensor

# def pixel_shuffle_conv2d(input: torch.Tensor, weight: torch.Tensor, bias=None, stride=1, padding=0, dilation=1, groups=1, upscale_factor=2) -> torch.Tensor:
#     x = F.conv2d(input, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
#     return F.pixel_shuffle(x, upscale_factor)

def test_pixel_shuffle_conv2d():
    results = {}
    
    # Test case 1: Basic test with default parameters
    input1 = torch.randn(1, 4, 8, 8, device='cuda')
    weight1 = torch.randn(16, 4, 3, 3, device='cuda')
    results["test_case_1"] = fused_pixel_shuffle_conv2d(input1, weight1)
    
    # Test case 2: Test with bias
    input2 = torch.randn(1, 4, 8, 8, device='cuda')
    weight2 = torch.randn(16, 4, 3, 3, device='cuda')
    bias2 = torch.randn(16, device='cuda')
    results["test_case_2"] = fused_pixel_shuffle_conv2d(input2, weight2, bias=bias2)
    
    # Test case 3: Test with stride
    input3 = torch.randn(1, 4, 16, 16, device='cuda')
    weight3 = torch.randn(16, 4, 3, 3, device='cuda')
    results["test_case_3"] = fused_pixel_shuffle_conv2d(input3, weight3, stride=2)
    
    # Test case 4: Test with padding
    input4 = torch.randn(1, 4, 8, 8, device='cuda')
    weight4 = torch.randn(16, 4, 3, 3, device='cuda')
    results["test_case_4"] = fused_pixel_shuffle_conv2d(input4, weight4, padding=1)

    for mode in ("standard", "outlier"):
        outs = []
        for _ in range(2):
            x = rand_tensor((2, 3, 16, 16), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            w = rand_tensor((32, 3, 3, 3), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            b = rand_tensor((32,), dtype=torch.float32, mode="standard")
            outs.append(fused_pixel_shuffle_conv2d(x, w, bias=b, padding=1, upscale_factor=2))
        results[f"test_random_{mode}"] = outs
    
    return results

test_results = test_pixel_shuffle_conv2d()
