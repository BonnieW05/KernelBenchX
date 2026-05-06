import torch

def sgd_step(param, grad, momentum_buffer=None, lr=0.1, momentum=0.9, weight_decay=0.0, dampening=0.0, nesterov=False):
    """
    Performs a single step of SGD optimization on a parameter tensor.
    
    Args:
        param (Tensor): Parameter tensor to update (in-place).
        grad (Tensor): Gradient tensor.
        momentum_buffer (Tensor, optional): Momentum buffer tensor. If None and momentum > 0,
                                           a new zero tensor will be created.
        lr (float): Learning rate. Default: 0.1
        momentum (float): Momentum factor. Default: 0.9
        weight_decay (float): Weight decay (L2 penalty). Default: 0.0
        dampening (float): Dampening for momentum. Default: 0.0
        nesterov (bool): Enables Nesterov momentum. Default: False
        
    Returns:
        tuple: Updated (param, momentum_buffer) or just param if momentum=0
    """
    if weight_decay != 0:
        grad = grad.add(param, alpha=weight_decay)
        
    if momentum > 0:
        if momentum_buffer is None:
            momentum_buffer = torch.zeros_like(grad)
            
        momentum_buffer.mul_(momentum).add_(grad, alpha=1 - dampening)
        
        if nesterov:
            grad = grad.add(momentum_buffer, alpha=momentum)
        else:
            grad = momentum_buffer
            
    param.add_(grad, alpha=-lr)
    
    if momentum > 0:
        return param, momentum_buffer
    else:
        return param, None

##################################################################################################################################################


import torch
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_tensor

def test_sgd_step():
    results = {}
    
    # Test case 1: Basic SGD update with momentum = 0
    param1 = torch.randn(128, device='cuda')
    grad1 = torch.randn(128, device='cuda')
    updated_param1 = sgd_step(param1.clone(), grad1, momentum=0.0)
    results["test_case_1"] = updated_param1
    
    # Test case 2: With momentum
    param2 = torch.randn(256, device='cuda')
    grad2 = torch.randn(256, device='cuda')
    momentum_buffer2 = torch.zeros(256, device='cuda')
    updated_param2, updated_buffer2 = sgd_step(param2.clone(), grad2, momentum_buffer2.clone())
    results["test_case_2"] = (updated_param2, updated_buffer2)
    
    # Test case 3: With weight decay
    param3 = torch.randn(128, device='cuda')
    grad3 = torch.randn(128, device='cuda')
    updated_param3 = sgd_step(param3.clone(), grad3, momentum=0.0, weight_decay=0.01)
    results["test_case_3"] = updated_param3
    
    # Test case 4: With Nesterov momentum
    param4 = torch.randn(128, device='cuda')
    grad4 = torch.randn(128, device='cuda')
    momentum_buffer4 = torch.zeros(128, device='cuda')
    updated_param4, updated_buffer4 = sgd_step(
        param4.clone(), grad4, momentum_buffer4.clone(), nesterov=True)
    results["test_case_4"] = (updated_param4, updated_buffer4)

    for mode in ("standard", "outlier"):
        outs = []
        for _ in range(3):
            p = rand_tensor((512,), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            g = rand_tensor((512,), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            mb = rand_tensor((512,), dtype=torch.float32, mode="standard")
            out_p, out_mb = sgd_step(p.clone(), g, mb.clone(), lr=0.05, momentum=0.9, weight_decay=0.01, dampening=0.0, nesterov=True)
            outs.append((out_p, out_mb))
        results[f"test_random_{mode}"] = outs
    
    return results

test_results = test_sgd_step()
