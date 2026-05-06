import torch
import torch.nn.functional as F

def fused_repeat_interleave_log_softmax(input, repeats, dim=None, *, output_size=None, dtype=None, out=None):
    repeated_input = torch.repeat_interleave(input, repeats, dim=dim)
    if dtype is not None:
        repeated_input = repeated_input.to(dtype)
    output = F.log_softmax(repeated_input, dim=dim, dtype=dtype)
    return output

##################################################################################################################################################


import torch
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_tensor, rand_int

# def fused_repeat_interleave_log_softmax(input, repeats, dim=None, *, output_size=None, dtype=None, out=None):
#     repeated_input = torch.repeat_interleave(input, repeats, dim=dim)
#     if dtype is not None:
#         repeated_input = repeated_input.to(dtype)
#     output = F.log_softmax(repeated_input, dim=dim, dtype=dtype)
#     return output

def test_fused_repeat_interleave_log_softmax():
    results = {}
    
    # Test case 1: Basic test with dim=None
    input1 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    repeats1 = 2
    results["test_case_1"] = fused_repeat_interleave_log_softmax(input1, repeats1)
    
    # Test case 2: Test with specified dim
    input2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    repeats2 = 2
    dim2 = 1
    results["test_case_2"] = fused_repeat_interleave_log_softmax(input2, repeats2, dim=dim2)
    
    # Test case 3: Test with dtype conversion
    input3 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    repeats3 = 3
    dtype3 = torch.float64
    results["test_case_3"] = fused_repeat_interleave_log_softmax(input3, repeats3, dtype=dtype3)
    
    # Test case 4: Test with specified dim and dtype conversion
    input4 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    repeats4 = 2
    dim4 = 0
    dtype4 = torch.float32
    results["test_case_4"] = fused_repeat_interleave_log_softmax(input4, repeats4, dim=dim4, dtype=dtype4)

    for mode in ("standard", "outlier"):
        outs = []
        for _ in range(2):
            x = rand_tensor((32, 128), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            outs.append(fused_repeat_interleave_log_softmax(x, 2, dim=1, dtype=torch.float32))
        for _ in range(2):
            x = rand_tensor((64,), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            r = rand_int((64,), low=1, high=4, dtype=torch.int64)
            outs.append(fused_repeat_interleave_log_softmax(x, r, dim=0, dtype=torch.float32))
        results[f"test_random_{mode}"] = outs
    
    return results

test_results = test_fused_repeat_interleave_log_softmax()
