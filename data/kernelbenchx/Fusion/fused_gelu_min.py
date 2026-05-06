import torch
import torch.nn.functional as F
import torch

def fused_gelu_min(input, approximate='none', dim=None, keepdim=False, out=None):
    if approximate == 'none':
        output = input * torch.erf(input / (2.0 ** 0.5)) / 2.0
    elif approximate == 'tanh':
        output = 0.5 * input * (1 + torch.tanh(((2.0 / torch.pi) ** 0.5) * (input + 0.044715 * input ** 3)))
    else:
        raise ValueError("Unknown approximation method. Choose either 'none' or 'tanh'.")
    if dim is None:
        return torch.min(output)
    else:
        (min_values, indices) = torch.min(output, dim=dim, keepdim=keepdim)
        if out is not None:
            out[0].copy_(min_values)
            out[1].copy_(indices)
        return (min_values, indices)

##################################################################################################################################################


import torch
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_tensor

def test_gelu_min():
    results = {}

    # Test case 1: Default approximate='none', no dim, no keepdim
    input_tensor = torch.tensor([0.5, -0.5, 1.0, -1.0], device='cuda')
    results['test_case_1'] = fused_gelu_min(input_tensor)

    # Test case 2: approximate='tanh', no dim, no keepdim
    input_tensor = torch.tensor([0.5, -0.5, 1.0, -1.0], device='cuda')
    results['test_case_2'] = fused_gelu_min(input_tensor, approximate='tanh')

    # Test case 3: approximate='none', with dim, no keepdim
    input_tensor = torch.tensor([[0.5, -0.5], [1.0, -1.0]], device='cuda')
    results['test_case_3'] = fused_gelu_min(input_tensor, dim=1)

    # Test case 4: approximate='tanh', with dim, keepdim=True
    input_tensor = torch.tensor([[0.5, -0.5], [1.0, -1.0]], device='cuda')
    results['test_case_4'] = fused_gelu_min(input_tensor, approximate='tanh', dim=1, keepdim=True)

    for mode in ("standard", "outlier"):
        outs = []
        for _ in range(2):
            x1 = rand_tensor((4096,), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            outs.append(fused_gelu_min(x1, approximate="tanh"))
        for _ in range(2):
            x2 = rand_tensor((64, 128), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            outs.append(fused_gelu_min(x2, approximate="none", dim=1, keepdim=False))
        results[f"test_random_{mode}"] = outs

    return results

test_results = test_gelu_min()
