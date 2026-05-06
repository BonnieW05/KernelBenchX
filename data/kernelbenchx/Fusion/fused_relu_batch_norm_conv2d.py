import torch
import torch.nn.functional as F

def fused_relu_batch_norm_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, running_mean=None, running_var=None, bn_weight=None, bn_bias=None, training=False, momentum=0.1, eps=1e-05, inplace=False):
    """
    Applies a 2D convolution over the input tensor, followed by batch normalization 
    and then applies the ReLU activation function element-wise to the normalized result.
    
    Args:
        input (Tensor): The input tensor of shape (minibatch, in_channels, iH, iW).
        weight (Tensor): The convolution filters of shape (out_channels, in_channels / groups, kH, kW).
        bias (Tensor, optional): Optional bias tensor of shape (out_channels). Default: None.
        stride (int or tuple, optional): The stride of the convolution kernel. Default: 1.
        padding (int, tuple, or string, optional): Padding added to all sides of the input. Default: 0.
        dilation (int or tuple, optional): The spacing between kernel elements. Default: 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1.
        running_mean (Tensor, optional): The running mean for batch normalization. Default: None.
        running_var (Tensor, optional): The running variance for batch normalization. Default: None.
        bn_weight (Tensor, optional): Learnable scaling factor for batch normalization (gamma). Default: None.
        bn_bias (Tensor, optional): Learnable shift factor for batch normalization (beta). Default: None.
        training (bool, optional): If True, updates running statistics for batch normalization. Default: False.
        momentum (float, optional): Value for updating the running mean and variance in batch normalization. Default: 0.1.
        eps (float, optional): A small value added for numerical stability in batch normalization. Default: 1e-5.
        inplace (bool, optional): If True, performs ReLU in-place. Default: False.
    
    Returns:
        Tensor: The output tensor after convolution, batch normalization, and ReLU activation.
    """
    conv_result = F.conv2d(input, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
    bn_result = F.batch_norm(conv_result, running_mean, running_var, bn_weight, bn_bias, training=training, momentum=momentum, eps=eps)
    return F.relu(bn_result, inplace=inplace)


##################################################################################################################################################


import torch
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_tensor


def test_relu_batch_norm_conv2d():
    results = {}

    input_tensor = torch.randn(4, 3, 32, 32, device="cuda", dtype=torch.float32)
    weight_tensor = torch.randn(6, 3, 3, 3, device="cuda", dtype=torch.float32)
    bias_tensor = torch.randn(6, device="cuda", dtype=torch.float32)

    running_mean = torch.zeros(6, device="cuda", dtype=torch.float32)
    running_var = torch.ones(6, device="cuda", dtype=torch.float32)
    bn_weight = torch.ones(6, device="cuda", dtype=torch.float32)
    bn_bias = torch.zeros(6, device="cuda", dtype=torch.float32)

    results["test_case_1"] = fused_relu_batch_norm_conv2d(
        input=input_tensor,
        weight=weight_tensor,
        bias=bias_tensor,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        running_mean=running_mean,
        running_var=running_var,
        bn_weight=bn_weight,
        bn_bias=bn_bias,
        training=True,
        momentum=0.1,
        eps=1e-5,
        inplace=False,
    )

    results["test_case_2"] = fused_relu_batch_norm_conv2d(
        input=input_tensor,
        weight=weight_tensor,
        bias=bias_tensor,
        stride=2,
        padding=1,
        dilation=1,
        groups=1,
        running_mean=running_mean,
        running_var=running_var,
        bn_weight=bn_weight,
        bn_bias=bn_bias,
        training=False,
        momentum=0.1,
        eps=1e-5,
        inplace=False,
    )

    for mode in ("standard", "outlier"):
        outs = []
        for _ in range(2):
            x = rand_tensor((2, 3, 32, 32), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            w = rand_tensor((8, 3, 3, 3), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            b = rand_tensor((8,), dtype=torch.float32, mode="standard")

            rm = torch.zeros(8, device="cuda", dtype=torch.float32)
            rv = torch.ones(8, device="cuda", dtype=torch.float32)
            bn_w = rand_tensor((8,), dtype=torch.float32, mode="standard")
            bn_b = rand_tensor((8,), dtype=torch.float32, mode="standard")

            outs.append(
                fused_relu_batch_norm_conv2d(
                    input=x,
                    weight=w,
                    bias=b,
                    stride=1,
                    padding=1,
                    dilation=1,
                    groups=1,
                    running_mean=rm,
                    running_var=rv,
                    bn_weight=bn_w,
                    bn_bias=bn_b,
                    training=True,
                    momentum=0.1,
                    eps=1e-5,
                    inplace=False,
                )
            )
        results[f"test_random_{mode}"] = outs

    return results


test_results = test_relu_batch_norm_conv2d()