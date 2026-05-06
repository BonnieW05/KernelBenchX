import torch
import torch.nn.functional as F
from torch import Tensor

def min_gelu(input: Tensor, dim=None, keepdim=False, approximate='none', out=None) -> Tensor:
    """
    Computes the minimum of the GELU activation of the input tensor along the specified dimension(s).
    
    Args:
        input (Tensor): The input tensor.
        dim (int, optional): The dimension to reduce. If None, returns the minimum of all elements.
        keepdim (bool, optional): Whether the output tensor retains :attr:`dim` as size 1. Default is False.
        approximate (str, optional): The approximation method for GELU. Default is 'none'.
                                      'none' computes exact GELU, 'tanh' computes the approximate GELU using the tanh method.
        out (Tensor, optional): The output tensor.

    Returns:
        Tensor: The minimum value after applying GELU.
        If dim is specified, returns a namedtuple (values, indices), otherwise returns the minimum value tensor.
    """
    if approximate == 'none':
        gelu_input = input * torch.erf(input / torch.sqrt(torch.tensor(2.0, device=input.device, dtype=input.dtype))) / 2.0
    elif approximate == 'tanh':
        gelu_input = 0.5 * input * (1 + torch.tanh(torch.sqrt(torch.tensor(2 / torch.pi, device=input.device, dtype=input.dtype)) * (input + 0.044715 * input ** 3)))
    else:
        raise ValueError(f"Invalid value for approximate: {approximate}. Choose 'none' or 'tanh'.")
    if dim is not None:
        return torch.min(gelu_input, dim=dim, keepdim=keepdim, out=out)
    else:
        return torch.min(gelu_input, out=out)

##################################################################################################################################################


import torch
import torch.nn.functional as F
from torch import Tensor
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_tensor

# def min_gelu(input: Tensor, dim=None, keepdim=False, approximate='none', out=None) -> Tensor:
#     """
#     Computes the minimum of the GELU activation of the input tensor along the specified dimension(s).
    
#     Args:
#         input (Tensor): The input tensor.
#         dim (int, optional): The dimension to reduce. If None, returns the minimum of all elements.
#         keepdim (bool, optional): Whether the output tensor retains :attr:`dim` as size 1. Default is False.
#         approximate (str, optional): The approximation method for GELU. Default is 'none'.
#                                       'none' computes exact GELU, 'tanh' computes the approximate GELU using the tanh method.
#         out (Tensor, optional): The output tensor.

#     Returns:
#         Tensor: The minimum value after applying GELU.
#         If dim is specified, returns a namedtuple (values, indices), otherwise returns the minimum value tensor.
#     """
#     if approximate == 'none':
#         gelu_input = input * torch.erf(input / torch.sqrt(torch.tensor(2.0))) / 2.0
#     elif approximate == 'tanh':
#         gelu_input = 0.5 * input * (1 + torch.tanh(torch.sqrt(torch.tensor(2 / torch.pi)) * (input + 0.044715 * input ** 3)))
#     else:
#         raise ValueError(f"Invalid value for approximate: {approximate}. Choose 'none' or 'tanh'.")
#     if dim is not None:
#         return torch.min(gelu_input, dim=dim, keepdim=keepdim, out=out)
#     else:
#         return torch.min(gelu_input, out=out)

def test_min_gelu():
    results = {}
    
    # Test case 1: Default parameters
    input_tensor = torch.tensor([1.0, -0.5, 0.0, 2.0], device='cuda')
    results["test_case_1"] = min_gelu(input_tensor)
    
    # Test case 2: With dimension reduction
    input_tensor = torch.tensor([[1.0, -0.5], [0.0, 2.0]], device='cuda')
    results["test_case_2"] = min_gelu(input_tensor, dim=1)
    
    # Test case 3: With dimension reduction and keepdim=True
    input_tensor = torch.tensor([[1.0, -0.5], [0.0, 2.0]], device='cuda')
    results["test_case_3"] = min_gelu(input_tensor, dim=1, keepdim=True)
    
    # Test case 4: Using 'tanh' approximation
    input_tensor = torch.tensor([1.0, -0.5, 0.0, 2.0], device='cuda')
    results["test_case_4"] = min_gelu(input_tensor, approximate='tanh')

    for mode in ("standard", "outlier"):
        outs = []
        x1 = rand_tensor((4096,), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
        outs.append(min_gelu(x1))
        x2 = rand_tensor((128, 256), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
        outs.append(min_gelu(x2, dim=1))
        outs.append(min_gelu(x2, dim=1, keepdim=True))
        outs.append(min_gelu(x1, approximate='tanh'))
        results[f"test_random_{mode}"] = outs
    
    return results

test_results = test_min_gelu()
