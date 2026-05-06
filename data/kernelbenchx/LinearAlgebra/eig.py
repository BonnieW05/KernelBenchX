import torch

def eig(A):
    (eigenvalues, eigenvectors) = torch.linalg.eig(A)
    return (eigenvalues, eigenvectors)

##################################################################################################################################################


import torch
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_tensor

# def eig(A):
#     (eigenvalues, eigenvectors) = torch.linalg.eig(A)
#     return (eigenvalues, eigenvectors)

def test_eig():
    results = {}

    # Test case 1: 2x2 matrix with distinct eigenvalues
    A1 = torch.tensor([[2.0, 0.0], [0.0, 3.0]], device='cuda')
    results["test_case_1"] = eig(A1)

    # Test case 2: 2x2 matrix with repeated eigenvalues
    A2 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device='cuda')
    results["test_case_2"] = eig(A2)

    # Test case 3: 3x3 matrix with complex eigenvalues
    A3 = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], device='cuda')
    results["test_case_3"] = eig(A3)

    # Test case 4: 3x3 matrix with real eigenvalues
    A4 = torch.tensor([[4.0, 1.0, 0.0], [1.0, 4.0, 0.0], [0.0, 0.0, 5.0]], device='cuda')
    results["test_case_4"] = eig(A4)

    for mode in ("standard", "outlier"):
        outs = []
        for n in (8, 16):
            x = rand_tensor((n, n), dtype=torch.float64, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            A = (x + x.mT) / 2
            A = A + torch.eye(n, device="cuda", dtype=torch.float64) * 1e-3
            outs.append(eig(A))
        results[f"test_random_{mode}"] = outs

    return results

test_results = test_eig()
