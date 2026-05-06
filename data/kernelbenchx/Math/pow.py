import torch

def pow(input_tensor, exponent, out=None):
    """
    This function mimics the behavior of torch.pow, which raises each element of the input tensor to the power of the exponent.
    
    Args:
    - input_tensor (Tensor): the input tensor.
    - exponent (float or Tensor): the exponent value, either a scalar or a tensor with the same number of elements as input_tensor.
    - out (Tensor, optional): the output tensor to store the result.
    
    Returns:
    - Tensor: The result of raising each element of the input_tensor to the power of the exponent.
    """
    return torch.pow(input_tensor, exponent, out=out)

##################################################################################################################################################


import torch
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_tensor

def test_pow():
    results = {}

    # Test case 1: input_tensor and exponent are scalars
    input_tensor = torch.tensor([2.0], device='cuda')
    exponent = 3.0
    results["test_case_1"] = pow(input_tensor, exponent)

    # Test case 2: input_tensor is a tensor, exponent is a scalar
    input_tensor = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    exponent = 2.0
    results["test_case_2"] = pow(input_tensor, exponent)

    # Test case 3: input_tensor and exponent are tensors of the same shape
    input_tensor = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    exponent = torch.tensor([3.0, 2.0, 1.0], device='cuda')
    results["test_case_3"] = pow(input_tensor, exponent)

    # Test case 4: input_tensor is a tensor, exponent is a negative scalar
    input_tensor = torch.tensor([4.0, 9.0, 16.0], device='cuda')
    exponent = -0.5
    results["test_case_4"] = pow(input_tensor, exponent)

    for mode in ("standard", "outlier"):
        outs = []
        x = rand_tensor((64, 64), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
        x = x.abs() + 1e-3
        outs.append(pow(x, 2.0))
        outs.append(pow(x, -0.5))
        results[f"test_random_{mode}"] = outs

    return results

test_results = test_pow()
