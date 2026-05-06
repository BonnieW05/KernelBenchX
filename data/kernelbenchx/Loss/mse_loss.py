import torch

def mse_loss(input, target, reduction='mean'):
    """
    Mean Squared Error loss between input and target.
    
    Args:
        input (Tensor): Predicted values of any shape
        target (Tensor): Ground truth values, same shape as input
        reduction (str): 'none' | 'mean' | 'sum'
        
    Returns:
        Tensor: Computed MSE loss
    """
    return torch.nn.functional.mse_loss(input, target, reduction=reduction)

##################################################################################################################################################


import torch
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_tensor

def test_mse_loss():
    results = {}

    # Test case 1: Basic 1D regression
    input1 = torch.randn(10, device='cuda')
    target1 = torch.randn(10, device='cuda')
    results["test_case_1"] = mse_loss(input1, target1)

    # Test case 2: With reduction='none' (per-element loss)
    input2 = torch.randn(3, 3, device='cuda')
    target2 = torch.randn(3, 3, device='cuda')
    results["test_case_2"] = mse_loss(input2, target2, reduction='none')

    # Test case 3: Image reconstruction (N, C, H, W)
    input3 = torch.randn(2, 3, 8, 8, device='cuda')
    target3 = torch.randn(2, 3, 8, 8, device='cuda')
    results["test_case_3"] = mse_loss(input3, target3)

    for mode in ("standard", "outlier"):
        for reduction in ("mean", "sum"):
            outs = []
            for _ in range(2):
                x = rand_tensor((128, 256), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
                y = rand_tensor((128, 256), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
                outs.append(mse_loss(x, y, reduction=reduction))
            results[f"test_random_{mode}_{reduction}"] = outs

    return results

test_results = test_mse_loss()
