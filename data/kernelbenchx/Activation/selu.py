import torch
import torch.nn.functional as F
import math

def selu(input: torch.Tensor, inplace: bool=False) -> torch.Tensor:
    """
    Applies the element-wise SELU (Scaled Exponential Linear Unit) function to the input tensor.
    
    The SELU function is defined as:
    SELU(x) = scale * (max(0, x) + min(0, alpha * (exp(x) - 1)))
    where alpha is approximately 1.673 and scale is approximately 1.051.
    
    Args:
    - input (torch.Tensor): The input tensor.
    - inplace (bool, optional): If set to True, will do the operation in-place. Default is False.

    Returns:
    - torch.Tensor: The resulting tensor after applying SELU function.
    """
    alpha = 1.6732632423543772
    scale = 1.0507009873554805
    return scale * (torch.maximum(input, torch.zeros_like(input)) + torch.minimum(input, alpha * (torch.exp(input) - 1)))

##################################################################################################################################################


import torch
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_tensor

def test_selu():
    # Initialize a dictionary to store test results
    results = {}

    # Test case 1: Positive values
    input_tensor_1 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    results["test_case_1"] = selu(input_tensor_1)

    # Test case 2: Negative values
    input_tensor_2 = torch.tensor([-1.0, -2.0, -3.0], device='cuda')
    results["test_case_2"] = selu(input_tensor_2)

    # Test case 3: Mixed values
    input_tensor_3 = torch.tensor([-1.0, 0.0, 1.0], device='cuda')
    results["test_case_3"] = selu(input_tensor_3)

    # Test case 4: Zero values
    input_tensor_4 = torch.tensor([0.0, 0.0, 0.0], device='cuda')
    results["test_case_4"] = selu(input_tensor_4)

    for mode in ("standard", "outlier"):
        outs = []
        for _ in range(3):
            x = rand_tensor((512, 512), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=20.0)
            outs.append(selu(x))
        results[f"test_random_{mode}"] = outs

    return results

test_results = test_selu()
