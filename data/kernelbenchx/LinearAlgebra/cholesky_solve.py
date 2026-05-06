import torch

def cholesky_solve(B, L, upper=False, out=None):
    """
    Solve a linear system given a Cholesky factorization of an SPD/Hermitian PD matrix.

    Args:
        B (Tensor): Right-hand side tensor of shape (*, n, k), where * are optional batch dimensions.
        L (Tensor): Cholesky factor of shape (*, n, n), either lower- or upper-triangular.
        upper (bool, optional): If True, `L` is interpreted as upper-triangular. Default: False (lower-triangular).
        out (Tensor, optional): Optional output tensor.

    Returns:
        Tensor: Solution tensor X with the same shape as B.
    """
    return torch.cholesky_solve(B, L, upper=upper, out=out)

##################################################################################################################################################


import torch
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_tensor

def test_cholesky_solve():
    results = {}

    # Test case 1: Lower triangular matrix
    B1 = torch.tensor([[1.0], [2.0]], device='cuda')
    L1 = torch.tensor([[2.0, 0.0], [1.0, 1.0]], device='cuda')
    results["test_case_1"] = cholesky_solve(B1, L1)

    # Test case 2: Upper triangular matrix
    B2 = torch.tensor([[1.0], [2.0]], device='cuda')
    L2 = torch.tensor([[2.0, 1.0], [0.0, 1.0]], device='cuda')
    results["test_case_2"] = cholesky_solve(B2, L2, upper=True)

    # Test case 3: Batch of matrices, lower triangular
    B3 = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]], device='cuda')
    L3 = torch.tensor([[[2.0, 0.0], [1.0, 1.0]], [[3.0, 0.0], [1.0, 2.0]]], device='cuda')
    results["test_case_3"] = cholesky_solve(B3, L3)

    # Test case 4: Batch of matrices, upper triangular
    B4 = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]], device='cuda')
    L4 = torch.tensor([[[2.0, 1.0], [0.0, 1.0]], [[3.0, 1.0], [0.0, 2.0]]], device='cuda')
    results["test_case_4"] = cholesky_solve(B4, L4, upper=True)

    for mode in ("standard", "outlier"):
        outs = []
        for batch, n, k in ((1, 16, 8), (4, 8, 4)):
            x = rand_tensor((batch, n, n), dtype=torch.float64, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            A = x @ x.mT
            A = (A + A.mT) / 2
            A = A + torch.eye(n, device="cuda", dtype=torch.float64) * 1e-3
            L = torch.linalg.cholesky(A)
            B = rand_tensor((batch, n, k), dtype=torch.float64, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            outs.append(cholesky_solve(B, L, upper=False))
            outs.append(cholesky_solve(B, L.mT, upper=True))
        results[f"test_random_{mode}"] = outs

    return results

test_results = test_cholesky_solve()
