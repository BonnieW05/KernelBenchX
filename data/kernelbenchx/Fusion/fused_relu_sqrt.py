import torch
from torch import Tensor


def fused_relu_sqrt(input: Tensor, inplace: bool=False, out: Tensor=None) -> Tensor:
    """
    Applies the rectified linear unit (ReLU) function to each element in input,
    and then computes the square root of the result.
    
    Args:
        input (Tensor): The input tensor.
        inplace (bool, optional): If True, modifies input in-place (if possible). Default is False.
        out (Tensor, optional): The output tensor.
    
    Returns:
        Tensor: The result of applying relu followed by sqrt.
    
    Example:
        >>> import torch
        >>> a = torch.tensor([-1.0, 0.0, 4.0, 9.0])
        >>> result = relu_sqrt(a)
        >>> print(result)
        tensor([0.0000, 0.0000, 2.0000, 3.0000])
        >>> result = relu_sqrt(a, inplace=True)
        >>> print(result)
        tensor([0.0000, 0.0000, 2.0000, 3.0000])
    """
    if input.dtype != torch.float32 and input.dtype != torch.float64:
        input = input.float()
    if inplace:
        input.relu_()
        input.sqrt_()
        return input
    elif out is not None:
        out.copy_(torch.sqrt(torch.relu(input)))
        return out
    else:
        return torch.sqrt(torch.relu(input))

##################################################################################################################################################


import torch
from torch import Tensor
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_int, rand_tensor

# def relu_sqrt(input: Tensor, inplace: bool=False, out: Tensor=None) -> Tensor:
#     if input.dtype != torch.float32 and input.dtype != torch.float64:
#         input = input.float()
#     if inplace:
#         input.relu_()
#         input.sqrt_()
#         return input
#     elif out is not None:
#         out.copy_(torch.sqrt(torch.relu(input)))
#         return out
#     else:
#         return torch.sqrt(torch.relu(input))

def test_relu_sqrt():
    results = {}
    
    # Test case 1: Default parameters
    a = torch.tensor([-1.0, 0.0, 4.0, 9.0], device='cuda')
    results["test_case_1"] = fused_relu_sqrt(a)
    
    # Test case 2: Inplace operation
    b = torch.tensor([-1.0, 0.0, 4.0, 9.0], device='cuda')
    results["test_case_2"] = fused_relu_sqrt(b, inplace=True)
    
    # Test case 3: Out parameter
    c = torch.tensor([-1.0, 0.0, 4.0, 9.0], device='cuda')
    out = torch.empty_like(c)
    results["test_case_3"] = fused_relu_sqrt(c, out=out)
    
    # Test case 4: Non-float input
    d = torch.tensor([-1, 0, 4, 9], device='cuda')
    results["test_case_4"] = fused_relu_sqrt(d)

    for mode in ("standard", "outlier"):
        outs = []
        for _ in range(3):
            x = rand_tensor((1024, 1024), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            outs.append(fused_relu_sqrt(x))
        results[f"test_random_{mode}"] = outs

    outs_int = []
    for _ in range(2):
        xi = rand_int((512, 512), low=-10, high=10, device="cuda", dtype=torch.int64)
        outs_int.append(fused_relu_sqrt(xi))
    results["test_random_int"] = outs_int
    
    return results

test_results = test_relu_sqrt()
