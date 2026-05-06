import torch

def ldl_factor(A, hermitian=False, out=None):
    """
    Perform the LDL factorization of a symmetric or Hermitian matrix.

    Args:
        A (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions consisting of symmetric or Hermitian matrices.
        hermitian (bool, optional): whether to consider the input to be Hermitian or symmetric. Default is False.
        out (tuple, optional): tuple of two tensors to write the output to. Ignored if None. Default is None.

    Returns:
        namedtuple: A named tuple `(LD, pivots)`. 
                    LD is the compact representation of L and D.
                    pivots is a tensor containing the pivot indices.
    """
    (LD, pivots) = torch.linalg.ldl_factor(A, hermitian=hermitian, out=out)
    return (LD, pivots)

##################################################################################################################################################


import torch
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_tensor

def test_ldl_factor():
    results = {}

    # Test case 1: Symmetric matrix
    A1 = torch.tensor([[4.0, 1.0], [1.0, 3.0]], device='cuda')
    results["test_case_1"] = ldl_factor(A1)

    # Test case 2: Hermitian matrix
    A2 = torch.tensor([[2.0, 1.0j], [-1.0j, 2.0]], device='cuda')
    results["test_case_2"] = ldl_factor(A2, hermitian=True)

    # Test case 3: Batch of symmetric matrices
    A3 = torch.tensor([[[4.0, 1.0], [1.0, 3.0]], [[2.0, 0.5], [0.5, 2.0]]], device='cuda')
    results["test_case_3"] = ldl_factor(A3)

    # Test case 4: Batch of Hermitian matrices
    A4 = torch.tensor([[[2.0, 1.0j], [-1.0j, 2.0]], [[3.0, 0.5j], [-0.5j, 3.0]]], device='cuda')
    results["test_case_4"] = ldl_factor(A4, hermitian=True)

    for mode in ("standard", "outlier"):
        outs = []
        x = rand_tensor((16, 16), dtype=torch.float64, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
        A = (x + x.mT) / 2
        A = A + torch.eye(16, device="cuda", dtype=torch.float64) * 1e-3
        outs.append(ldl_factor(A, hermitian=False))

        xc = rand_tensor((16, 16), dtype=torch.complex64, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
        Ah = (xc + xc.conj().mT) / 2
        Ah = Ah + torch.eye(16, device="cuda", dtype=torch.complex64) * 1e-3
        outs.append(ldl_factor(Ah, hermitian=True))
        results[f"test_random_{mode}"] = outs

    return results

test_results = test_ldl_factor()
