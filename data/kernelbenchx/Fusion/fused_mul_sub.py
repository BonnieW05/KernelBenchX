import torch

def fused_mul_sub(input, other_mul, other_sub, alpha=1, out=None):
    """
    Multiplies the input tensor by another tensor or number, then subtracts another tensor or number from the result,
    scaled by a given alpha. This operation is performed element-wise.

    Args:
        input (Tensor): The input tensor to be multiplied.
        other_mul (Tensor or Number): The tensor or number to multiply with `input`.
        other_sub (Tensor or Number): The tensor or number to subtract from the multiplication result.
        alpha (Number, optional): The multiplier for :attr:`other_sub`. Default is 1.
        out (Tensor, optional): The output tensor.

    Returns:
        Tensor: The result of the operation.
    """
    result = input * other_mul - alpha * other_sub
    if out is not None:
        out.copy_(result)
        return out
    return result

##################################################################################################################################################


import torch
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_tensor

def test_mul_sub():
    results = {}

    # Test case 1: input, other_mul, other_sub are tensors
    input_tensor = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    other_mul_tensor = torch.tensor([0.5, 1.5, 2.5], device='cuda')
    other_sub_tensor = torch.tensor([0.1, 0.2, 0.3], device='cuda')
    results["test_case_1"] = fused_mul_sub(input_tensor, other_mul_tensor, other_sub_tensor)

    # Test case 2: input is a tensor, other_mul is a number, other_sub is a tensor
    other_mul_number = 2.0
    results["test_case_2"] = fused_mul_sub(input_tensor, other_mul_number, other_sub_tensor)

    # Test case 3: input is a tensor, other_mul is a tensor, other_sub is a number
    other_sub_number = 0.5
    results["test_case_3"] = fused_mul_sub(input_tensor, other_mul_tensor, other_sub_number)

    # Test case 4: input, other_mul, other_sub are numbers
    input_number = 3.0
    results["test_case_4"] = fused_mul_sub(input_number, other_mul_number, other_sub_number)

    for mode in ("standard", "outlier"):
        outs = []
        for _ in range(3):
            x = rand_tensor((512, 512), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            y = rand_tensor((512, 512), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            z = rand_tensor((512, 512), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            outs.append(fused_mul_sub(x, y, z, alpha=-0.5))
        results[f"test_random_{mode}"] = outs

    return results

test_results = test_mul_sub()
