import torch
import torch.nn.functional as F

def fused_cos_avg_pool1d(input: torch.Tensor, kernel_size: int, stride: int=None, padding: int=0, ceil_mode: bool=False, count_include_pad: bool=True) -> torch.Tensor:
    """
    Applies the cosine function element-wise to the input tensor, followed by 1D average pooling.

    Args:
        input (Tensor): The input tensor of shape (minibatch, in_channels, iW).
        kernel_size (int): Size of the pooling window.
        stride (int, optional): Stride of the pooling window. Defaults to `kernel_size`.
        padding (int, optional): Zero-padding added to both sides of the input. Default is 0.
        ceil_mode (bool, optional): If True, uses ceil instead of floor to compute the output shape. Default is False.
        count_include_pad (bool, optional): If True, includes the zero-padding in the averaging calculation. Default is True.

    Returns:
        Tensor: The resulting tensor after cosine transformation and 1D average pooling.
    """
    cos_input = torch.cos(input)
    return F.avg_pool1d(cos_input, kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode, count_include_pad=count_include_pad)

##################################################################################################################################################


import torch
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_tensor

# def cos_avg_pool1d(input: torch.Tensor, kernel_size: int, stride: int=None, padding: int=0, ceil_mode: bool=False, count_include_pad: bool=True) -> torch.Tensor:
#     cos_input = torch.cos(input)
#     return F.avg_pool1d(cos_input, kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode, count_include_pad=count_include_pad)

def test_cos_avg_pool1d():
    results = {}

    # Test case 1: Basic functionality with default parameters
    input_tensor_1 = torch.tensor([[[0.0, 1.0, 2.0, 3.0, 4.0]]], device='cuda')
    results['test_case_1'] = fused_cos_avg_pool1d(input_tensor_1, kernel_size=2)

    # Test case 2: Custom stride
    input_tensor_2 = torch.tensor([[[0.0, 1.0, 2.0, 3.0, 4.0]]], device='cuda')
    results['test_case_2'] = fused_cos_avg_pool1d(input_tensor_2, kernel_size=2, stride=1)

    # Test case 3: With padding
    input_tensor_3 = torch.tensor([[[0.0, 1.0, 2.0, 3.0, 4.0]]], device='cuda')
    results['test_case_3'] = fused_cos_avg_pool1d(input_tensor_3, kernel_size=2, padding=1)

    # Test case 4: Using ceil_mode
    input_tensor_4 = torch.tensor([[[0.0, 1.0, 2.0, 3.0, 4.0]]], device='cuda')
    results['test_case_4'] = fused_cos_avg_pool1d(input_tensor_4, kernel_size=2, ceil_mode=True)

    for mode in ("standard", "outlier"):
        outs = []
        for _ in range(3):
            x = rand_tensor((4, 16, 256), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            outs.append(fused_cos_avg_pool1d(x, kernel_size=4, stride=2, padding=1, ceil_mode=False))
        results[f"test_random_{mode}"] = outs

    return results

test_results = test_cos_avg_pool1d()
