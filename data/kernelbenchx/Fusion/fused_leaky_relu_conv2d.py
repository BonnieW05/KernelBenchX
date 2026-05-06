import torch
import torch.nn.functional as F
from torch import Tensor

def fused_leaky_relu_conv2d(input: Tensor, weight: Tensor, bias: Tensor=None, stride: int=1, padding: int=0, dilation: int=1, groups: int=1, negative_slope: float=0.01, inplace: bool=False) -> Tensor:
    """
    Applies a 2D convolution over the input tensor, followed by applying the Leaky ReLU
    activation function element-wise to the result.

    Args:
        input (Tensor): The input tensor to apply the convolution to.
        weight (Tensor): The weight tensor for the convolution.
        bias (Tensor, optional): The bias tensor for the convolution.
        stride (int, optional): The stride of the convolution. Default is 1.
        padding (int, optional): The padding applied to the input. Default is 0.
        dilation (int, optional): The dilation of the convolution. Default is 1.
        groups (int, optional): The number of groups for the convolution. Default is 1.
        negative_slope (float, optional): The negative slope for the Leaky ReLU function. Default is 0.01.
        inplace (bool, optional): If True, will perform the operation in-place. Default is False.

    Returns:
        Tensor: The output tensor after applying convolution and Leaky ReLU activation.
    """
    conv_output = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
    output = F.leaky_relu(conv_output, negative_slope, inplace)
    return output

##################################################################################################################################################


import torch
import torch.nn.functional as F
from torch import Tensor
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_tensor

# def leaky_relu_conv2d(input: Tensor, weight: Tensor, bias: Tensor=None, stride: int=1, padding: int=0, dilation: int=1, groups: int=1, negative_slope: float=0.01, inplace: bool=False) -> Tensor:
#     conv_output = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
#     output = F.leaky_relu(conv_output, negative_slope, inplace)
#     return output

def test_leaky_relu_conv2d():
    results = {}
    
    # Test case 1: Basic test with default parameters
    input = torch.randn(1, 3, 32, 32, device='cuda')
    weight = torch.randn(6, 3, 3, 3, device='cuda')
    bias = torch.randn(6, device='cuda')
    results["test_case_1"] = fused_leaky_relu_conv2d(input, weight, bias)
    
    # Test case 2: Test with stride and padding
    input = torch.randn(1, 3, 32, 32, device='cuda')
    weight = torch.randn(6, 3, 3, 3, device='cuda')
    bias = torch.randn(6, device='cuda')
    results["test_case_2"] = fused_leaky_relu_conv2d(input, weight, bias, stride=2, padding=1)
    
    # Test case 3: Test with dilation and groups
    input = torch.randn(1, 6, 32, 32, device='cuda')
    weight = torch.randn(6, 1, 3, 3, device='cuda')
    bias = torch.randn(6, device='cuda')
    results["test_case_3"] = fused_leaky_relu_conv2d(input, weight, bias, dilation=2, groups=6)
    
    # Test case 4: Test with negative_slope and inplace
    input = torch.randn(1, 3, 32, 32, device='cuda')
    weight = torch.randn(6, 3, 3, 3, device='cuda')
    bias = torch.randn(6, device='cuda')
    results["test_case_4"] = fused_leaky_relu_conv2d(input, weight, bias, negative_slope=0.1, inplace=True)

    for mode in ("standard", "outlier"):
        outs = []
        for _ in range(2):
            x = rand_tensor((2, 3, 32, 32), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            w = rand_tensor((8, 3, 3, 3), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            b = rand_tensor((8,), dtype=torch.float32, mode="standard")
            outs.append(fused_leaky_relu_conv2d(x, w, b, stride=1, padding=1, negative_slope=0.1, inplace=False))
        results[f"test_random_{mode}"] = outs
    
    return results

test_results = test_leaky_relu_conv2d()
