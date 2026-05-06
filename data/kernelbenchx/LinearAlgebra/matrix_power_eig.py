import torch

def matrix_power_eig(A: torch.Tensor, k: float, *, out: torch.Tensor=None) -> torch.Tensor:
    """
    Computes the matrix power A^k of a square matrix A using eigendecomposition.
    
    Args:
        A (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions consisting of square matrices.
        k (float or complex): the exponent to which the matrix A is to be raised.
        out (Tensor, optional): output tensor. Ignored if `None`. Default: `None`.

    Returns:
        Tensor: the matrix A raised to the power k.
    
    The function uses the eigendecomposition to compute A^k as V diag(Λ^k) V^(-1),
    where Λ are the eigenvalues and V the eigenvectors of A. The result may be complex 
    even if A is real due to complex eigenvalues. 
    
    Warning:
        If A is not diagonalizable, the result may not be accurate. Gradients might be numerically unstable 
        if the distance between any two eigenvalues is close to zero.
    """
    (eigvals, eigvecs) = torch.linalg.eigh(A)
    eigvals_power_k = torch.pow(eigvals, k)
    A_power_k = torch.matmul(eigvecs, torch.matmul(torch.diag(eigvals_power_k), eigvecs.transpose(-1, -2)))
    if out is not None:
        out.copy_(A_power_k)
        return out
    return A_power_k

##################################################################################################################################################


import torch
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_tensor

def test_matrix_power_eig():
    results = {}

    # Test case 1: Simple 2x2 matrix with integer exponent
    A1 = torch.tensor([[2.0, 0.0], [0.0, 3.0]], device='cuda')
    k1 = 2
    results["test_case_1"] = matrix_power_eig(A1, k1)

    # Test case 2: 3x3 matrix with fractional exponent
    A2 = torch.randn(3, 3, device='cuda')
    A2 = A2 @ A2.mT + torch.eye(3, device='cuda') * 1e-3
    k2 = 0.5
    results["test_case_2"] = matrix_power_eig(A2, k2)

    # Test case 4: Batch of 2x2 matrices with integer exponent
    A4 = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], device='cuda')
    k4 = 3
    results["test_case_4"] = matrix_power_eig(A4, k4)

    for mode in ("standard", "outlier"):
        outs = []
        for n, k in ((16, 0.5), (16, 2.0)):
            x = rand_tensor((n, n), dtype=torch.float64, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            A = x @ x.mT + torch.eye(n, device="cuda", dtype=torch.float64) * 1e-3
            outs.append(matrix_power_eig(A, k))
        results[f"test_random_{mode}"] = outs

    return results

test_results = test_matrix_power_eig()
