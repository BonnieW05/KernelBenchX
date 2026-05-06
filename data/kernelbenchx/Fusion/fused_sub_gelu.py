import torch
import torch.nn.functional as F


def fused_sub_gelu(input, other, alpha=1, approximate='none', out=None):
    """
    Subtracts 'other', scaled by 'alpha', from 'input', and then applies the Gaussian Error Linear Units (GELU)
    activation function to the result.

    Args:
        input (Tensor): The input tensor.
        other (Tensor or Number): The tensor or number to subtract from input.
        alpha (Number, optional): The multiplier for other. Default is 1.
        approximate (str, optional): The approximation method for GELU. Default is 'none'.
        out (Tensor, optional): The output tensor.

    Returns:
        Tensor: The result of applying GELU activation after subtraction.
    """
    result = input - alpha * other
    if approximate == 'tanh':
        result = 0.5 * result * (1 + torch.tanh(((2.0 / torch.pi) ** 0.5) * (result + 0.044715 * result ** 3)))
    else:
        result = result * torch.erf(result / (2.0 ** 0.5)) * 0.5
    if out is not None:
        out.copy_(result)
        return out
    return result

##################################################################################################################################################


import torch
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_tensor

def test_sub_gelu():
    results = {}

    # Test case 1: Basic subtraction and GELU with default approximate
    input_tensor = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    other_tensor = torch.tensor([0.5, 1.0, 1.5], device='cuda')
    results["test_case_1"] = fused_sub_gelu(input_tensor, other_tensor)

    # Test case 2: Subtraction with alpha and GELU with default approximate
    alpha = 0.5
    results["test_case_2"] = fused_sub_gelu(input_tensor, other_tensor, alpha=alpha)

    # Test case 3: Subtraction and GELU with 'tanh' approximation
    approximate = 'tanh'
    results["test_case_3"] = fused_sub_gelu(input_tensor, other_tensor, approximate=approximate)

    # Test case 4: Subtraction with alpha and GELU with 'tanh' approximation
    results["test_case_4"] = fused_sub_gelu(input_tensor, other_tensor, alpha=alpha, approximate=approximate)

    for mode in ("standard", "outlier"):
        outs = []
        for _ in range(3):
            x = rand_tensor((4096,), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            y = rand_tensor((4096,), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            outs.append(fused_sub_gelu(x, y, alpha=0.5, approximate="tanh"))
        results[f"test_random_{mode}"] = outs

    return results

test_results = test_sub_gelu()
