import torch
import torch.nn.functional as F

def fused_gelu_std(input, dim=None, keepdim=False, correction=1, approximate='none', out=None):
    gelu_result = F.gelu(input, approximate=approximate)
    return torch.std(gelu_result, dim=dim, keepdim=keepdim, correction=correction, out=out)

##################################################################################################################################################


import torch
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_tensor

# def gelu_std(input, dim=None, keepdim=False, correction=1, approximate='none', out=None):
#     gelu_result = F.gelu(input, approximate=approximate)
#     return torch.std(gelu_result, dim=dim, keepdim=keepdim, correction=correction, out=out)

def test_gelu_std():
    results = {}
    
    # Test case 1: Default parameters
    input1 = torch.randn(10, device='cuda')
    results["test_case_1"] = fused_gelu_std(input1)
    
    # Test case 2: With dim parameter
    input2 = torch.randn(10, 20, device='cuda')
    results["test_case_2"] = fused_gelu_std(input2, dim=1)
    
    # Test case 3: With keepdim=True
    input3 = torch.randn(10, 20, device='cuda')
    results["test_case_3"] = fused_gelu_std(input3, dim=1, keepdim=True)
    
    # Test case 4: With approximate='tanh'
    input4 = torch.randn(10, device='cuda')
    results["test_case_4"] = fused_gelu_std(input4, approximate='tanh')

    for mode in ("standard", "outlier"):
        outs = []
        for dim in (None, 0, 1):
            x = rand_tensor((128, 256), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            outs.append(fused_gelu_std(x, dim=dim, keepdim=False, correction=1, approximate="tanh"))
        results[f"test_random_{mode}"] = outs
    
    return results

test_results = test_gelu_std()
