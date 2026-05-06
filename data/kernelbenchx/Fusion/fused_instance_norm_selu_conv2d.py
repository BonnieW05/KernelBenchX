import torch
import torch.nn.functional as F
from torch import nn

def fused_instance_norm_selu_conv2d(input: torch.Tensor, weight: torch.Tensor, bias=None, stride=1, padding=0, dilation=1, groups=1, num_features=None, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False) -> torch.Tensor:
    conv_output = torch.nn.functional.conv2d(input, weight, bias, stride, padding, dilation, groups)
    selu_output = torch.nn.functional.selu(conv_output)
    normalized_output = torch.nn.functional.instance_norm(selu_output, eps=eps, momentum=momentum)
    return normalized_output

##################################################################################################################################################


import torch
import torch.nn.functional as F
from torch import nn
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_tensor

# def fused_instance_norm_selu_conv2d(input: torch.Tensor, weight: torch.Tensor, bias=None, stride=1, padding=0, dilation=1, groups=1, num_features=None, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False) -> torch.Tensor:
#     conv_output = torch.nn.functional.conv2d(input, weight, bias, stride, padding, dilation, groups)
#     selu_output = torch.nn.functional.selu(conv_output)
#     normalized_output = torch.nn.functional.instance_norm(selu_output, eps=eps, momentum=momentum)
#     return normalized_output

def test_fused_instance_norm_selu_conv2d():
    results = {}
    
    # Test case 1: Basic test with default parameters
    input_tensor = torch.randn(1, 3, 5, 5, device='cuda')
    weight_tensor = torch.randn(3, 3, 3, 3, device='cuda')
    results["test_case_1"] = fused_instance_norm_selu_conv2d(input_tensor, weight_tensor)
    
    # Test case 2: Test with stride
    results["test_case_2"] = fused_instance_norm_selu_conv2d(input_tensor, weight_tensor, stride=2)
    
    # Test case 3: Test with padding
    results["test_case_3"] = fused_instance_norm_selu_conv2d(input_tensor, weight_tensor, padding=1)

    for mode in ("standard", "outlier"):
        outs = []
        for _ in range(2):
            x = rand_tensor((2, 3, 32, 32), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            w = rand_tensor((8, 3, 3, 3), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            b = rand_tensor((8,), dtype=torch.float32, mode="standard")
            outs.append(fused_instance_norm_selu_conv2d(x, w, bias=b, padding=1))
        results[f"test_random_{mode}"] = outs
    
    return results

test_results = test_fused_instance_norm_selu_conv2d()
