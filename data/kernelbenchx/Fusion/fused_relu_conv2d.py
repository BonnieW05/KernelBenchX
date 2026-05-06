import torch
import torch.nn.functional as F

def fused_relu_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, inplace=False):
    """
    Applies a 2D convolution followed by a ReLU activation function.

    Args:
        input (Tensor): The input tensor of shape (minibatch, in_channels, iH, iW).
        weight (Tensor): The convolution filters of shape (out_channels, in_channels / groups, kH, kW).
        bias (Tensor, optional): Optional bias tensor of shape (out_channels). Default: None.
        stride (int or tuple, optional): The stride of the convolution kernel. Default: 1.
        padding (int, tuple, or string, optional): Padding added to all sides of the input. Default: 0.
        dilation (int or tuple, optional): The spacing between kernel elements. Default: 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1.
        inplace (bool, optional): If True, will perform ReLU operation in-place. Default: False.

    Returns:
        Tensor: A tensor resulting from the 2D convolution followed by ReLU activation.
    """
    conv_out = F.conv2d(input, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
    return F.relu(conv_out, inplace=inplace)

##################################################################################################################################################


import torch
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_tensor

# def relu_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, inplace=False):
#     """
#     Applies a 2D convolution followed by a ReLU activation function.

#     Args:
#         input (Tensor): The input tensor of shape (minibatch, in_channels, iH, iW).
#         weight (Tensor): The convolution filters of shape (out_channels, in_channels / groups, kH, kW).
#         bias (Tensor, optional): Optional bias tensor of shape (out_channels). Default: None.
#         stride (int or tuple, optional): The stride of the convolution kernel. Default: 1.
#         padding (int, tuple, or string, optional): Padding added to all sides of the input. Default: 0.
#         dilation (int or tuple, optional): The spacing between kernel elements. Default: 1.
#         groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1.
#         inplace (bool, optional): If True, will perform ReLU operation in-place. Default: False.

#     Returns:
#         Tensor: A tensor resulting from the 2D convolution followed by ReLU activation.
#     """
#     conv_out = F.conv2d(input, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
#     return F.relu(conv_out, inplace=inplace)

def test_relu_conv2d():
    results = {}
    
    # Test case 1: Basic test with default parameters
    input1 = torch.randn(1, 3, 5, 5, device='cuda')
    weight1 = torch.randn(2, 3, 3, 3, device='cuda')
    results["test_case_1"] = fused_relu_conv2d(input1, weight1)
    
    # Test case 2: Test with bias
    input2 = torch.randn(1, 3, 5, 5, device='cuda')
    weight2 = torch.randn(2, 3, 3, 3, device='cuda')
    bias2 = torch.randn(2, device='cuda')
    results["test_case_2"] = fused_relu_conv2d(input2, weight2, bias=bias2)
    
    # Test case 3: Test with stride and padding
    input3 = torch.randn(1, 3, 5, 5, device='cuda')
    weight3 = torch.randn(2, 3, 3, 3, device='cuda')
    results["test_case_3"] = fused_relu_conv2d(input3, weight3, stride=2, padding=1)
    
    # Test case 4: Test with dilation and groups
    input4 = torch.randn(1, 4, 5, 5, device='cuda')
    weight4 = torch.randn(2, 2, 3, 3, device='cuda')
    results["test_case_4"] = fused_relu_conv2d(input4, weight4, dilation=2, groups=2)

    for mode in ("standard", "outlier"):
        outs = []
        for _ in range(2):
            x = rand_tensor((2, 3, 32, 32), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            w = rand_tensor((8, 3, 3, 3), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            b = rand_tensor((8,), dtype=torch.float32, mode="standard")
            outs.append(fused_relu_conv2d(x, w, bias=b, stride=1, padding=1))
        results[f"test_random_{mode}"] = outs
    
    return results

test_results = test_relu_conv2d()
