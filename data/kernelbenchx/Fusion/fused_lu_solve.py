import torch

def fused_lu_solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Solves the linear system Ax = b using LU decomposition.
    
    Args:
        A (torch.Tensor): The input matrix of shape (n, n).
        b (torch.Tensor): The right-hand side tensor of shape (n,).
        
    Returns:
        torch.Tensor: The solution tensor x of shape (n,).
    """
    # LU decomposition of matrix A
    P, L, U = torch.linalg.lu(A)
    # Solve for x using L and U from LU decomposition
    x = torch.linalg.solve(L @ U, b)
    return x

##################################################################################################################################################


import torch
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_tensor

def test_fused_lu_solve():
    results = {}
    
    # Test case 1: Simple 2x2 system
    A1 = torch.tensor([[3.0, 1.0], [1.0, 2.0]], device='cuda')
    b1 = torch.tensor([9.0, 8.0], device='cuda')
    results["test_case_1"] = fused_lu_solve(A1, b1)
    
    # Test case 2: 3x3 system
    A2 = torch.tensor([[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]], device='cuda')
    b2 = torch.tensor([6.0, 4.0, 3.0], device='cuda')
    results["test_case_2"] = fused_lu_solve(A2, b2)
    
    # Test case 3: 4x4 system
    A3 = torch.tensor([[4.0, 3.0, 2.0, 1.0], [3.0, 2.0, 1.0, 4.0], [2.0, 1.0, 4.0, 3.0], [1.0, 4.0, 3.0, 2.0]], device='cuda')
    b3 = torch.tensor([10.0, 11.0, 12.0, 13.0], device='cuda')
    results["test_case_3"] = fused_lu_solve(A3, b3)
    
    # Test case 4: Singular matrix (should raise an error)
    A4 = torch.tensor([[1.0, 2.0], [2.0, 4.0]], device='cuda')
    b4 = torch.tensor([5.0, 10.0], device='cuda')
    try:
        results["test_case_4"] = fused_lu_solve(A4, b4)
    except RuntimeError as e:
        results["test_case_4"] = str(e)

    for mode in ("standard", "outlier"):
        outs = []
        for _ in range(2):
            n = 64
            x = rand_tensor((n, n), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            A = x @ x.T + 1e-2 * torch.eye(n, device="cuda", dtype=torch.float32)
            b = rand_tensor((n,), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            outs.append(fused_lu_solve(A, b))
        results[f"test_random_{mode}"] = outs
    
    return results

test_results = test_fused_lu_solve()
