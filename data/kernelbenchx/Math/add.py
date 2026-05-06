import torch

def add(input, other, alpha=1, out=None):
    """
    Adds the tensor or number 'other', scaled by 'alpha', to the 'input' tensor.
    
    Args:
        input (Tensor): The input tensor.
        other (Tensor or Number): The tensor or number to add to input.
        alpha (Number, optional): The multiplier for 'other'. Default is 1.
        out (Tensor, optional): The output tensor. If provided, the result will be stored in this tensor.
        
    Returns:
        Tensor: The result of adding 'other' scaled by 'alpha' to 'input'.
    """
    return torch.add(input, other, alpha=alpha, out=out)

##################################################################################################################################################


import torch
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_tensor

def test_add():
    results = {}

    # Test case 1: Adding two tensors with default alpha
    input1 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    other1 = torch.tensor([4.0, 5.0, 6.0], device='cuda')
    results["test_case_1"] = add(input1, other1)

    # Test case 2: Adding a tensor and a scalar with default alpha
    input2 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    other2 = 2.0
    results["test_case_2"] = add(input2, other2)

    # Test case 3: Adding two tensors with a specified alpha
    input3 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    other3 = torch.tensor([4.0, 5.0, 6.0], device='cuda')
    results["test_case_3"] = add(input3, other3, alpha=0.5)

    # Test case 4: Adding a tensor and a scalar with a specified alpha
    input4 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    other4 = 2.0
    results["test_case_4"] = add(input4, other4, alpha=0.5)

    for mode in ("standard", "outlier"):
        outs = []
        for _ in range(3):
            x = rand_tensor((256, 256), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            y = rand_tensor((256, 256), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            outs.append(add(x, y))
        results[f"test_random_{mode}"] = outs

    return results

test_results = test_add()
