import torch
import torch.nn.functional as F


def combined_activation(input, weight1, weight2, bias, *, out=None):
    """
    Perform the combined activation function which includes matrix multiplication,
    sigmoid, tanh, element-wise multiplication, and addition.

    Args:
        input (Tensor): Input tensor of shape (*, N, D_in), where * denotes any batch dimensions.
        weight1 (Tensor): Weight matrix of shape (D_in, D_out).
        weight2 (Tensor): Weight tensor for element-wise multiplication, must be broadcastable 
                          to the shape of the intermediate activation.
        bias (Tensor): Bias tensor, must be broadcastable to the shape of the output.
        out (Tensor, optional): Output tensor to store the result, ignored if None.

    Returns:
        Tensor: Output tensor of shape (*, N, D_out).
    """
    z = torch.mm(input, weight1)
    s = torch.sigmoid(z)
    t = torch.tanh(s)
    m = t * weight2
    y = m + bias
    if out is not None:
        out.copy_(y)
        return out
    return y

##################################################################################################################################################


import torch
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_tensor

# def combined_activation(input, weight1, weight2, bias, *, out=None):
#     """
#     Perform the combined activation function which includes matrix multiplication,
#     sigmoid, tanh, element-wise multiplication, and addition.

#     Args:
#         input (Tensor): Input tensor of shape (*, N, D_in), where * denotes any batch dimensions.
#         weight1 (Tensor): Weight matrix of shape (D_in, D_out).
#         weight2 (Tensor): Weight tensor for element-wise multiplication, must be broadcastable 
#                           to the shape of the intermediate activation.
#         bias (Tensor): Bias tensor, must be broadcastable to the shape of the output.
#         out (Tensor, optional): Output tensor to store the result, ignored if None.

#     Returns:
#         Tensor: Output tensor of shape (*, N, D_out).
#     """
#     z = torch.mm(input, weight1)
#     s = torch.sigmoid(z)
#     t = torch.tanh(s)
#     m = t * weight2
#     y = m + bias
#     if out is not None:
#         out.copy_(y)
#         return out
#     return y

def test_combined_activation():
    results = {}

    # Test case 1
    input1 = torch.randn(2, 3, device='cuda')
    weight1_1 = torch.randn(3, 4, device='cuda')
    weight2_1 = torch.randn(2, 4, device='cuda')
    bias1 = torch.randn(2, 4, device='cuda')
    results["test_case_1"] = combined_activation(input1, weight1_1, weight2_1, bias1)

    # Test case 2
    input2 = torch.randn(3, 3, device='cuda')
    weight1_2 = torch.randn(3, 5, device='cuda')
    weight2_2 = torch.randn(3, 5, device='cuda')
    bias2 = torch.randn(3, 5, device='cuda')
    results["test_case_2"] = combined_activation(input2, weight1_2, weight2_2, bias2)

    # Test case 3
    input3 = torch.randn(4, 3, device='cuda')
    weight1_3 = torch.randn(3, 6, device='cuda')
    weight2_3 = torch.randn(4, 6, device='cuda')
    bias3 = torch.randn(4, 6, device='cuda')
    results["test_case_3"] = combined_activation(input3, weight1_3, weight2_3, bias3)

    # Test case 4
    input4 = torch.randn(5, 3, device='cuda')
    weight1_4 = torch.randn(3, 7, device='cuda')
    weight2_4 = torch.randn(5, 7, device='cuda')
    bias4 = torch.randn(5, 7, device='cuda')
    results["test_case_4"] = combined_activation(input4, weight1_4, weight2_4, bias4)

    for mode in ("standard", "outlier"):
        outs = []
        for _ in range(2):
            m, din, dout = 256, 128, 192
            x = rand_tensor((m, din), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=20.0)
            w1 = rand_tensor((din, dout), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=20.0)
            w2 = rand_tensor((m, dout), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=20.0)
            b = rand_tensor((m, dout), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=20.0)
            outs.append(combined_activation(x, w1, w2, b))
        results[f"test_random_{mode}"] = outs

    return results

test_results = test_combined_activation()
