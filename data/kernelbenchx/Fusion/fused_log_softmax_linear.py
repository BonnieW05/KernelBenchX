import torch
import torch.nn.functional as F

def fused_log_softmax_linear(input, weight, bias=None, dim=-1, dtype=None):
    """
    Applies a linear transformation to the input tensor followed by the log_softmax activation function.
    
    Args:
        input (Tensor): The input tensor of shape `(*, in_features)`.
        weight (Tensor): The weight matrix of shape `(out_features, in_features)`.
        bias (Tensor, optional): The optional bias tensor of shape `(out_features)`. Default: None.
        dim (int, optional): The dimension along which log_softmax will be computed. Default: -1.
        dtype (torch.dtype, optional): The desired data type of the returned tensor.
        
    Returns:
        Tensor: The output tensor after applying the linear transformation followed by log_softmax.
    """
    output = torch.matmul(input, weight.T)
    if bias is not None:
        output += bias
    return F.log_softmax(output, dim=dim, dtype=dtype)

##################################################################################################################################################


import torch
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_tensor

# def log_softmax_linear(input, weight, bias=None, dim=-1, dtype=None):
#     output = torch.matmul(input, weight.T)
#     if bias is not None:
#         output += bias
#     return F.log_softmax(output, dim=dim, dtype=dtype)

def test_log_softmax_linear():
    results = {}

    # Test case 1: Basic test with bias
    input1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    weight1 = torch.tensor([[0.5, 0.5], [0.5, -0.5]], device='cuda')
    bias1 = torch.tensor([0.1, -0.1], device='cuda')
    results["test_case_1"] = fused_log_softmax_linear(input1, weight1, bias1)

    # Test case 2: Test without bias
    input2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    weight2 = torch.tensor([[0.5, 0.5], [0.5, -0.5]], device='cuda')
    results["test_case_2"] = fused_log_softmax_linear(input2, weight2)

    # Test case 3: Test with different dim
    input3 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    weight3 = torch.tensor([[0.5, 0.5], [0.5, -0.5]], device='cuda')
    bias3 = torch.tensor([0.1, -0.1], device='cuda')
    results["test_case_3"] = fused_log_softmax_linear(input3, weight3, bias3, dim=0)

    # Test case 4: Test with dtype
    input4 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    weight4 = torch.tensor([[0.5, 0.5], [0.5, -0.5]], device='cuda')
    bias4 = torch.tensor([0.1, -0.1], device='cuda')
    results["test_case_4"] = fused_log_softmax_linear(input4, weight4, bias4, dtype=torch.float64)

    for mode in ("standard", "outlier"):
        outs = []
        for _ in range(2):
            x = rand_tensor((32, 128), dtype=torch.float16, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            w = rand_tensor((256, 128), dtype=torch.float16, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            b = rand_tensor((256,), dtype=torch.float16, mode="standard")
            outs.append(fused_log_softmax_linear(x, w, b, dim=-1))
        results[f"test_random_{mode}"] = outs

    return results

test_results = test_log_softmax_linear()
