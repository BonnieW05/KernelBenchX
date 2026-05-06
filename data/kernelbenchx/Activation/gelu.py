import torch
import torch.nn.functional as F

def gelu(input: torch.Tensor, approximate: str='none') -> torch.Tensor:
    return F.gelu(input, approximate=approximate)

##################################################################################################################################################


import torch
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_tensor

# def gelu(input: torch.Tensor, approximate: str='none') -> torch.Tensor:
#     return F.gelu(input, approximate=approximate)

def test_gelu():
    results = {}
    
    # Test case 1: Default approximate='none'
    input_tensor_1 = torch.tensor([-1.0, 0.0, 1.0], device='cuda')
    results["test_case_1"] = gelu(input_tensor_1)
    
    # Test case 2: approximate='tanh'
    input_tensor_2 = torch.tensor([-1.0, 0.0, 1.0], device='cuda')
    results["test_case_2"] = gelu(input_tensor_2, approximate='tanh')
    
    # Test case 3: Larger tensor with default approximate='none'
    input_tensor_3 = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device='cuda')
    results["test_case_3"] = gelu(input_tensor_3)
    
    # Test case 4: Larger tensor with approximate='tanh'
    input_tensor_4 = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device='cuda')
    results["test_case_4"] = gelu(input_tensor_4, approximate='tanh')

    for mode in ("standard", "outlier"):
        outs_none = []
        outs_tanh = []
        for _ in range(3):
            x = rand_tensor((1024, 1024), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            outs_none.append(gelu(x, approximate='none'))
            outs_tanh.append(gelu(x, approximate='tanh'))
        results[f"test_random_{mode}_none"] = outs_none
        results[f"test_random_{mode}_tanh"] = outs_tanh
    
    return results

test_results = test_gelu()
