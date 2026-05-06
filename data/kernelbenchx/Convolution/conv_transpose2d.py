import torch
import torch.nn.functional as F

def conv_transpose2d(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor=None, stride: int=1, padding: int=0, output_padding: int=0, groups: int=1, dilation: int=1) -> torch.Tensor:
    """Applies a 2D transposed convolution operator over an input image composed of several input planes.

    Args:
        input (torch.Tensor): Input tensor of shape (minibatch, in_channels, iH, iW).
        weight (torch.Tensor): Filters tensor of shape (in_channels, out_channels / groups, kH, kW).
        bias (torch.Tensor, optional): Bias tensor of shape (out_channels). Default: None.
        stride (int or tuple, optional): Stride of the transposed convolution. Default: 1.
        padding (int or tuple, optional): Padding added to both sides of the input. Default: 0.
        output_padding (int or tuple, optional): Additional size added to one side of each dimension in the output shape. Default: 0.
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1.
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1.

    Returns:
        torch.Tensor: Output tensor after applying the transposed convolution.
    """
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    if isinstance(output_padding, int):
        output_padding = (output_padding, output_padding)
    return F.conv_transpose2d(input, weight, bias, stride, padding, output_padding, groups, dilation)

##################################################################################################################################################


import torch
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_tensor

def test_conv_transpose2d():
    results = {}

    # Test case 1: Basic transposed convolution with default parameters
    input1 = torch.randn(1, 4, 8, 8, device='cuda')
    weight1 = torch.randn(4, 6, 3, 3, device='cuda')
    results["test_case_1"] = conv_transpose2d(input1, weight1)

    # Test case 2: Transposed convolution with stride, padding and output_padding
    input2 = torch.randn(1, 4, 8, 8, device='cuda')
    weight2 = torch.randn(4, 6, 3, 3, device='cuda')
    bias2 = torch.randn(6, device='cuda')
    results["test_case_2"] = conv_transpose2d(input2, weight2, bias=bias2, stride=2, padding=1, output_padding=1)

    # Test case 3: Grouped transposed convolution
    input3 = torch.randn(2, 4, 8, 8, device='cuda')
    weight3 = torch.randn(4, 2, 3, 3, device='cuda')
    results["test_case_3"] = conv_transpose2d(input3, weight3, groups=2)

    for mode in ("standard", "outlier"):
        outs = []
        for _ in range(2):
            x = rand_tensor((2, 8, 16, 16), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=20.0)
            w = rand_tensor((8, 6, 3, 3), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=20.0)
            outs.append(conv_transpose2d(x, w, stride=2, padding=1, output_padding=1))
        results[f"test_random_{mode}"] = outs

    return results


test_results = test_conv_transpose2d()
