import torch

def fused_hstack_div(tensors, divisor, *, rounding_mode=None, out=None):
    """
    Performs a fused operation combining horizontal stacking (hstack) and element-wise division.

    Args:
        tensors (sequence of Tensors): Sequence of tensors to be horizontally stacked.
                                        The tensors must have compatible shapes for stacking.
        divisor (Tensor or Number): The tensor or number to divide the stacked tensor by.
                                    Must be broadcastable to the shape of the stacked tensor.
        rounding_mode (str, optional): Type of rounding applied to the result. Options:
                                       'None', 'trunc', 'floor'. Default: None.
        out (Tensor, optional): Output tensor. Ignored if None. Default: None.

    Returns:
        Tensor: The result of stacking the tensors horizontally and dividing element-wise by the divisor.
    """
    X = torch.hstack(tensors)
    Y = torch.div(X, divisor, rounding_mode=rounding_mode)
    if out is not None:
        out.copy_(Y)
        return out
    return Y

##################################################################################################################################################


import torch
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_tensor

def test_fused_hstack_div():
    results = {}

    # Test case 1: Basic functionality with two tensors and a scalar divisor
    tensors1 = [torch.tensor([1, 2], device='cuda'), torch.tensor([3, 4], device='cuda')]
    divisor1 = 2
    results["test_case_1"] = fused_hstack_div(tensors1, divisor1)

    # Test case 3: Using rounding_mode='floor'
    tensors3 = [torch.tensor([1.5, 2.5], device='cuda'), torch.tensor([3.5, 4.5], device='cuda')]
    divisor3 = 2
    results["test_case_3"] = fused_hstack_div(tensors3, divisor3, rounding_mode='floor')

    # Test case 4: Using rounding_mode='trunc'
    tensors4 = [torch.tensor([1.5, 2.5], device='cuda'), torch.tensor([3.5, 4.5], device='cuda')]
    divisor4 = 2
    results["test_case_4"] = fused_hstack_div(tensors4, divisor4, rounding_mode='trunc')

    for mode in ("standard", "outlier"):
        outs = []
        for _ in range(2):
            a = rand_tensor((128, 64), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            b = rand_tensor((128, 64), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            outs.append(fused_hstack_div([a, b], 2.0))
        for _ in range(2):
            a = rand_tensor((128, 64), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            b = rand_tensor((128, 64), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            outs.append(fused_hstack_div([a, b], 2.0, rounding_mode="floor"))
        results[f"test_random_{mode}"] = outs

    return results

test_results = test_fused_hstack_div()
