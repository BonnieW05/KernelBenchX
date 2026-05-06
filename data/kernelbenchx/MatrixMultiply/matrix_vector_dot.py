import torch
from torch import Tensor

def matrix_vector_dot(A: Tensor, x: Tensor, y: Tensor, alpha: float, beta: float) -> Tensor:
    """
    Computes the matrix-vector product y = alpha * torch.mv(A, x) + beta * y
    and returns the dot product of the updated y and x.
    
    Args:
        A (Tensor): The input matrix of shape `(n, m)`.
        x (Tensor): The input vector of shape `(m,)`.
        y (Tensor): The target vector to be modified, of shape `(n,)`.
        alpha (float): Scalar multiplier for `torch.mv(A, x)`.
        beta (float): Scalar multiplier for `y`.
        
    Returns:
        Tensor: The dot product of the updated y and x.
    """
    y_new = alpha * torch.mv(A, x) + beta * y
    y.copy_(y_new)
    result = torch.dot(y, x)
    return result

##################################################################################################################################################


import torch
from torch import Tensor
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_tensor

def test_matrix_vector_dot():
    results = {}
    
    # Test case 1
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    x = torch.tensor([1.0, 1.0], device='cuda')
    y = torch.tensor([0.0, 0.0], device='cuda')
    alpha = 1.0
    beta = 0.0
    results["test_case_1"] = matrix_vector_dot(A, x, y, alpha, beta).item()
    
    # Test case 2
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    x = torch.tensor([1.0, 1.0], device='cuda')
    y = torch.tensor([1.0, 1.0], device='cuda')
    alpha = 1.0
    beta = 1.0
    results["test_case_2"] = matrix_vector_dot(A, x, y, alpha, beta).item()
    
    # Test case 3
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    x = torch.tensor([2.0, 3.0], device='cuda')
    y = torch.tensor([1.0, 1.0], device='cuda')
    alpha = 0.5
    beta = 0.5
    results["test_case_3"] = matrix_vector_dot(A, x, y, alpha, beta).item()
    
    # Test case 4
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    x = torch.tensor([1.0, 1.0], device='cuda')
    y = torch.tensor([2.0, 2.0], device='cuda')
    alpha = 2.0
    beta = 0.5
    results["test_case_4"] = matrix_vector_dot(A, x, y, alpha, beta).item()

    for mode in ("standard", "outlier"):
        outs = []
        for n, alpha, beta in ((64, 1.0, 0.0), (128, 0.5, 0.5)):
            A = rand_tensor((n, n), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            x = rand_tensor((n,), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            y = rand_tensor((n,), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            outs.append(matrix_vector_dot(A, x, y, alpha, beta))
        results[f"test_random_{mode}"] = outs
    
    return results

test_results = test_matrix_vector_dot()
