import torch
import torch.nn.functional as F

def fused_add_gelu(input, other, alpha=1, approximate='none', out=None):
    """
    Adds the tensor or number `other`, scaled by the multiplier `alpha`, to the input tensor `input`,
    and then applies the Gaussian Error Linear Units (GELU) activation function to the result.
    
    Args:
        input (Tensor): The input tensor.
        other (Tensor or Number): The tensor or number to add to input.
        alpha (Number, optional): The multiplier for other. Default is 1.
        approximate (str, optional): The approximation method for GELU. Default is 'none'.
        out (Tensor, optional): The output tensor.

    Returns:
        Tensor: The result of the operation.
    """
    result = input + alpha * other
    if approximate == 'none':
        result = F.gelu(result)
    elif approximate == 'tanh':
        result = 0.5 * result * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (result + 0.044715 * result ** 3)))
    else:
        raise ValueError("Invalid value for 'approximate'. Expected 'none' or 'tanh'.")
    return result

##################################################################################################################################################


import torch
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_tensor

def test_add_gelu():
    results = {}

    # Test case 1: Basic test with default parameters
    input_tensor = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    other_tensor = torch.tensor([0.5, 0.5, 0.5], device='cuda')
    results["test_case_1"] = fused_add_gelu(input_tensor, other_tensor)

    # Test case 2: Test with alpha parameter
    alpha = 2
    results["test_case_2"] = fused_add_gelu(input_tensor, other_tensor, alpha=alpha)

    # Test case 3: Test with approximate='tanh'
    approximate = 'tanh'
    results["test_case_3"] = fused_add_gelu(input_tensor, other_tensor, approximate=approximate)

    # Test case 4: Test with a scalar 'other'
    other_scalar = 0.5
    results["test_case_4"] = fused_add_gelu(input_tensor, other_scalar)

    for mode in ("standard", "outlier"):
        outs = []
        for _ in range(3):
            x = rand_tensor((512, 512), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            y = rand_tensor((512, 512), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            outs.append(fused_add_gelu(x, y, alpha=0.5, approximate="tanh"))
        results[f"test_random_{mode}"] = outs

    return results

test_results = test_add_gelu()
