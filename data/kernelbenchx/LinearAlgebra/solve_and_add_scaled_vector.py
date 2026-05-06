import torch


def solve_and_add_scaled_vector(A: torch.Tensor, b: torch.Tensor, y: torch.Tensor, alpha: float) -> torch.Tensor:
    x = torch.linalg.solve_triangular(A, b, upper=True)
    x += alpha * y
    return x

##################################################################################################################################################


import torch
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_tensor

def test_solve_and_add_scaled_vector():
    results = {}

    # Test case 1: Basic test with 2x2 upper triangular matrix
    A1 = torch.tensor([[2.0, 1.0], [0.0, 3.0]], device='cuda')
    b1 = torch.tensor([[5.0, 6.0], [7.0, 8]], device='cuda')
    y1 = torch.tensor([[1.0], [2.0]], device='cuda')
    alpha1 = 0.5
    results["test_case_1"] = solve_and_add_scaled_vector(A1, b1, y1, alpha1)

    for mode in ("standard", "outlier"):
        outs = []
        for n in (8, 16):
            A = rand_tensor((n, n), dtype=torch.float64, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            A = torch.triu(A)
            A = A + torch.eye(n, device="cuda", dtype=torch.float64) * 3.0
            b = rand_tensor((n, 4), dtype=torch.float64, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            y = rand_tensor((n, 1), dtype=torch.float64, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            outs.append(solve_and_add_scaled_vector(A, b, y, alpha=0.1))
        results[f"test_random_{mode}"] = outs
    return results

test_results = test_solve_and_add_scaled_vector()
