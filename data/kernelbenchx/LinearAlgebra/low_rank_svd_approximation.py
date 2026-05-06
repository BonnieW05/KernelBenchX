import torch

def low_rank_svd_approximation(A, k, *, full_matrices=True, out=None):
    """
    Computes a rank-k approximation of a matrix using its Singular Value Decomposition (SVD).

    Args:
        A (Tensor): Tensor of shape `(*, m, n)` where `*` is zero or more batch dimensions.
        k (int): Rank of the approximation (must satisfy `1 <= k <= min(m, n)`).
        full_matrices (bool, optional): Controls whether to compute the full or reduced SVD. Default: `True`.
        out (Tensor, optional): Output tensor. Ignored if `None`. Default: `None`.

    Returns:
        Tensor: The rank-k approximation of A.
    """
    (U, S, Vh) = torch.linalg.svd(A, full_matrices=full_matrices)
    U_k = U[..., :k]
    S_k = S[..., :k]
    Vh_k = Vh[..., :k, :]
    S_k_diag = torch.diag_embed(S_k)
    A_k = U_k @ S_k_diag @ Vh_k
    if out is not None:
        out.copy_(A_k)
        return out
    return A_k

##################################################################################################################################################


import torch
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_tensor

def test_low_rank_svd_approximation():
    results = {}

    # Test case 1: Basic rank-k approximation with full_matrices=True
    A = torch.randn(5, 4, device='cuda')
    k = 2
    results["test_case_1"] = low_rank_svd_approximation(A, k)

    # Test case 2: Basic rank-k approximation with full_matrices=False
    A = torch.randn(6, 3, device='cuda')
    k = 2
    results["test_case_2"] = low_rank_svd_approximation(A, k, full_matrices=False)

    # Test case 3: Batch matrix with full_matrices=True
    A = torch.randn(2, 5, 4, device='cuda')
    k = 3
    results["test_case_3"] = low_rank_svd_approximation(A, k)

    # Test case 4: Batch matrix with full_matrices=False
    A = torch.randn(3, 6, 3, device='cuda')
    k = 2
    results["test_case_4"] = low_rank_svd_approximation(A, k, full_matrices=False)

    for mode in ("standard", "outlier"):
        outs = []
        A = rand_tensor((32, 16), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
        outs.append(low_rank_svd_approximation(A, 8, full_matrices=True))
        outs.append(low_rank_svd_approximation(A, 8, full_matrices=False))
        Ab = rand_tensor((4, 32, 16), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
        outs.append(low_rank_svd_approximation(Ab, 8, full_matrices=False))
        results[f"test_random_{mode}"] = outs

    return results

test_results = test_low_rank_svd_approximation()
