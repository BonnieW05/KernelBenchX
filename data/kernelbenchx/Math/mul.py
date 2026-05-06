import torch

def mul(input, other, out=None):
    """
    Multiplies the input tensor by another tensor or a number, supporting broadcasting to a common shape,
    type promotion, and integer, float, and complex inputs.

    Parameters:
    - input (Tensor): The input tensor.
    - other (Tensor or Number): The tensor or number to multiply input by.
    - out (Tensor, optional): The output tensor.

    Returns:
    - Tensor: The result of the multiplication.
    """
    return torch.mul(input, other, out=out)

##################################################################################################################################################


import torch
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_tensor

def test_mul():
    results = {}

    # Test case 1: Multiply two tensors with broadcasting
    input1 = torch.tensor([1, 2, 3], device='cuda')
    other1 = torch.tensor([[1], [2], [3]], device='cuda')
    results["test_case_1"] = mul(input1, other1)

    # Test case 2: Multiply tensor by a scalar
    input2 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    other2 = 2.5
    results["test_case_2"] = mul(input2, other2)

    # Test case 3: Multiply complex tensors
    input3 = torch.tensor([1+2j, 3+4j], device='cuda')
    other3 = torch.tensor([5+6j, 7+8j], device='cuda')
    results["test_case_3"] = mul(input3, other3)

    # Test case 4: Multiply integer tensor by a float tensor
    input4 = torch.tensor([1, 2, 3], device='cuda')
    other4 = torch.tensor([0.5, 1.5, 2.5], device='cuda')
    results["test_case_4"] = mul(input4, other4)

    for mode in ("standard", "outlier"):
        outs = []
        for shape in ((1024,), (64, 64)):
            x = rand_tensor(shape, dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            y = rand_tensor(shape, dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            outs.append(mul(x, y))
        results[f"test_random_{mode}"] = outs

    return results

test_results = test_mul()
