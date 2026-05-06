import torch

def pseudoinverse_svd(A, full_matrices=True, rcond=1e-15, out=None):
    U, S, Vh = torch.linalg.svd(A, full_matrices=full_matrices)
    # Invert singular values larger than rcond * max(S)
    cutoff = rcond * S.max(dim=-1, keepdim=True).values
    S_inv = torch.where(S > cutoff, 1 / S, torch.zeros_like(S))
    # Create diagonal matrix of inverted singular values
    S_inv_mat = torch.diag_embed(S_inv)
    # Compute pseudoinverse
    A_pinv = Vh.transpose(-2, -1).conj() @ S_inv_mat @ U.transpose(-2, -1).conj()
    if out is not None:
        out.copy_(A_pinv)
        return out
    return A_pinv

##################################################################################################################################################


import torch
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_tensor

def test_pseudoinverse_svd():
    results = {}

    # Test case 1: Square matrix
    A1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    results["test_case_1"] = pseudoinverse_svd(A1)

    # Test case 4: Singular matrix
    A4 = torch.tensor([[1.0, 2.0], [2.0, 4.0]], device='cuda')
    results["test_case_4"] = pseudoinverse_svd(A4)

    for mode in ("standard", "outlier"):
        outs = []
        A = rand_tensor((16, 16), dtype=torch.float64, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
        outs.append(pseudoinverse_svd(A, full_matrices=True))
        Ar = rand_tensor((32, 16), dtype=torch.float64, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
        outs.append(pseudoinverse_svd(Ar, full_matrices=False))
        B = rand_tensor((16, 4), dtype=torch.float64, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
        A_low_rank = B @ B.mT
        outs.append(pseudoinverse_svd(A_low_rank, full_matrices=False))
        results[f"test_random_{mode}"] = outs

    return results

test_results = test_pseudoinverse_svd()
