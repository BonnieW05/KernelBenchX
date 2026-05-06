import torch

def det(A):
    return torch.linalg.det(A)

##################################################################################################################################################


import torch
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_tensor

# def det(A):
#     return torch.linalg.det(A)

def test_det():
    results = {}
    
    # Test case 1: 2x2 identity matrix
    A1 = torch.eye(2, device='cuda')
    results["test_case_1"] = det(A1).item()
    
    # Test case 2: 3x3 matrix with random values
    A2 = torch.rand((3, 3), device='cuda')
    results["test_case_2"] = det(A2).item()
    
    # Test case 3: 4x4 matrix with all zeros
    A3 = torch.zeros((4, 4), device='cuda')
    results["test_case_3"] = det(A3).item()
    
    # Test case 4: 2x2 matrix with specific values
    A4 = torch.tensor([[4.0, 7.0], [2.0, 6.0]], device='cuda')
    results["test_case_4"] = det(A4).item()

    for mode in ("standard", "outlier"):
        outs = []
        for _ in range(3):
            A = rand_tensor((4, 4), dtype=torch.float32, mode=mode, outlier_prob=0.02, outlier_scale=20.0)
            outs.append(det(A))
        results[f"test_random_{mode}"] = outs
    
    return results

test_results = test_det()
