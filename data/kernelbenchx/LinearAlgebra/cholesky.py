import torch

def cholesky(A: torch.Tensor, upper: bool=False, out: torch.Tensor=None) -> torch.Tensor:
    """
    Computes the Cholesky decomposition of a complex Hermitian or real symmetric positive-definite matrix.
    
    Args:
        A (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions
                    consisting of symmetric or Hermitian positive-definite matrices.
        upper (bool, optional): whether to return an upper triangular matrix.
                                Default is False, which means return a lower triangular matrix.
        out (Tensor, optional): output tensor. Ignored if `None`.
                                Default: `None`.
    
    Returns:
        Tensor: Cholesky decomposition of the input matrix.
    
    Example:
        >>> A = torch.randn(2, 2, dtype=torch.complex128)
        >>> A = A @ A.T.conj() + torch.eye(2)
        >>> L = cholesky_decomposition(A)
        >>> torch.dist(L @ L.T.conj(), A)
        tensor(4.4692e-16, dtype=torch.float64)
    """
    if not torch.allclose(A, A.mT) and (not torch.allclose(A, A.conj().mT)):
        raise RuntimeError('Input matrix is not Hermitian (resp. symmetric) positive-definite.')
    L = torch.linalg.cholesky(A, upper=upper, out=out)
    return L

##################################################################################################################################################


import torch
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_tensor

def test_cholesky():
    results = {}
    
    # Test case 1: Real symmetric positive-definite matrix, lower triangular
    A1 = torch.randn(2, 2, device='cuda', dtype=torch.float64)
    A1 = A1 @ A1.T + torch.eye(2, device='cuda', dtype=torch.float64)
    L1 = cholesky(A1)
    results["test_case_1"] = L1
    
    # Test case 2: Real symmetric positive-definite matrix, upper triangular
    A2 = torch.randn(2, 2, device='cuda', dtype=torch.float64)
    A2 = A2 @ A2.T + torch.eye(2, device='cuda', dtype=torch.float64)
    L2 = cholesky(A2, upper=True)
    results["test_case_2"] = L2
    
    # Test case 3: Complex Hermitian positive-definite matrix, lower triangular
    A3 = torch.randn(2, 2, device='cuda', dtype=torch.complex128)
    A3 = A3 @ A3.T.conj() + torch.eye(2, device='cuda', dtype=torch.complex128)
    L3 = cholesky(A3)
    results["test_case_3"] = L3
    
    # Test case 4: Complex Hermitian positive-definite matrix, upper triangular
    A4 = torch.randn(2, 2, device='cuda', dtype=torch.complex128)
    A4 = A4 @ A4.T.conj() + torch.eye(2, device='cuda', dtype=torch.complex128)
    L4 = cholesky(A4, upper=True)
    results["test_case_4"] = L4

    for mode in ("standard", "outlier"):
        outs = []
        for n, dtype in ((16, torch.float64), (16, torch.complex64)):
            x = rand_tensor((n, n), dtype=dtype, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            if dtype.is_complex:
                A = x @ x.conj().mT
                A = (A + A.conj().mT) / 2
            else:
                A = x @ x.mT
                A = (A + A.mT) / 2
            A = A + torch.eye(n, device="cuda", dtype=dtype) * 1e-3
            outs.append(cholesky(A, upper=False))
            outs.append(cholesky(A, upper=True))
        results[f"test_random_{mode}"] = outs
    
    return results

test_results = test_cholesky()
