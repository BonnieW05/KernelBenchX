import torch

def fftn(input, s=None, dim=None, norm=None, out=None):
    return torch.fft.fftn(input, s=s, dim=dim, norm=norm)

##################################################################################################################################################


import torch
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_tensor

# def fftn(input, s=None, dim=None, norm=None, out=None):
#     return torch.fft.fftn(input, s=s, dim=dim, norm=norm)

def test_fftn():
    results = {}
    
    # Test case 1: Only input tensor
    input_tensor = torch.randn(4, 4, device='cuda')
    results["test_case_1"] = fftn(input_tensor)
    
    # Test case 2: Input tensor with s parameter
    input_tensor = torch.randn(4, 4, device='cuda')
    s = (2, 2)
    results["test_case_2"] = fftn(input_tensor, s=s)
    
    # Test case 3: Input tensor with dim parameter
    input_tensor = torch.randn(4, 4, device='cuda')
    dim = (0, 1)
    results["test_case_3"] = fftn(input_tensor, dim=dim)
    
    # Test case 4: Input tensor with norm parameter
    input_tensor = torch.randn(4, 4, device='cuda')
    norm = "ortho"
    results["test_case_4"] = fftn(input_tensor, norm=norm)

    for mode in ("standard", "outlier"):
        outs = []
        x = rand_tensor((16, 16), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
        outs.append(fftn(x))
        outs.append(fftn(x, s=(8, 8)))
        outs.append(fftn(x, dim=(0, 1), norm="ortho"))
        results[f"test_random_{mode}"] = outs
    
    return results

test_results = test_fftn()
