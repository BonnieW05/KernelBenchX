import torch
import torch.nn.functional as F

def fused_mul_add_logsoftmax_dropout_bmm(input1, input2, other, mat2, p=0.5, training=True, inplace=False, dim=-1, *, out=None):
    """
    Performs a fused operation combining element-wise multiplication, addition,
    log-softmax activation, dropout, and batch matrix multiplication.
    
    Args:
        input1 (Tensor): The first input tensor.
        input2 (Tensor): The second input tensor.
        other (Tensor): A tensor or scalar to add to the result of element-wise multiplication.
        mat2 (Tensor): A tensor for batch matrix multiplication after dropout.
        p (float): The dropout probability.
        training (bool): Whether to apply dropout (only applies when True).
        inplace (bool): Whether to apply the operation in-place.
        dim (int): The dimension along which to apply log-softmax.
        out (Tensor, optional): If given, the result will be stored in this tensor.
        
    Returns:
        Tensor: The result of the fused operation.
    """
    Z = torch.mul(input1, input2)
    S = torch.add(Z, other)
    L = torch.nn.functional.log_softmax(S, dim=dim)
    D = torch.nn.functional.dropout(L, p=p, training=training, inplace=inplace)
    Y = torch.bmm(D, mat2)
    if out is not None:
        out.copy_(Y)
        return out
    return Y

##################################################################################################################################################


import torch
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_tensor

def test_fused_mul_add_logsoftmax_dropout_bmm():
    results = {}

    # Test case 1: Basic functionality
    input1 = torch.rand(2, 3, 4, device='cuda')
    input2 = torch.rand(2, 3, 4, device='cuda')
    other = torch.rand(2, 3, 4, device='cuda')
    mat2 = torch.rand(2, 4, 5, device='cuda')
    results["test_case_1"] = fused_mul_add_logsoftmax_dropout_bmm(input1, input2, other, mat2)

    # Test case 2: Different dropout probability
    input1 = torch.rand(2, 3, 4, device='cuda')
    input2 = torch.rand(2, 3, 4, device='cuda')
    other = torch.rand(2, 3, 4, device='cuda')
    mat2 = torch.rand(2, 4, 5, device='cuda')
    results["test_case_2"] = fused_mul_add_logsoftmax_dropout_bmm(input1, input2, other, mat2, p=0.3)

    # Test case 3: In-place operation
    input1 = torch.rand(2, 3, 4, device='cuda')
    input2 = torch.rand(2, 3, 4, device='cuda')
    other = torch.rand(2, 3, 4, device='cuda')
    mat2 = torch.rand(2, 4, 5, device='cuda')
    results["test_case_3"] = fused_mul_add_logsoftmax_dropout_bmm(input1, input2, other, mat2, inplace=True)

    # Test case 4: Different dimension for log-softmax
    input1 = torch.rand(2, 3, 4, device='cuda')
    input2 = torch.rand(2, 3, 4, device='cuda')
    other = torch.rand(2, 3, 4, device='cuda')
    mat2 = torch.rand(2, 4, 5, device='cuda')
    results["test_case_4"] = fused_mul_add_logsoftmax_dropout_bmm(input1, input2, other, mat2, dim=1)

    for mode in ("standard", "outlier"):
        outs = []
        for training in (False, True):
            x1 = rand_tensor((4, 16, 64), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            x2 = rand_tensor((4, 16, 64), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            o = rand_tensor((4, 16, 64), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            m2 = rand_tensor((4, 64, 32), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            outs.append(
                fused_mul_add_logsoftmax_dropout_bmm(
                    x1,
                    x2,
                    o,
                    m2,
                    p=0.1,
                    training=training,
                    inplace=False,
                    dim=-1,
                )
            )
        results[f"test_random_{mode}"] = outs

    return results

test_results = test_fused_mul_add_logsoftmax_dropout_bmm()
