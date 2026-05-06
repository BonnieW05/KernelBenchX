import torch

def lion_step(param, grad, exp_avg, lr=1e-4, beta1=0.9, beta2=0.99, weight_decay=0.0):
    """Lion optimizer step.

    Args:
        param (Tensor): Parameter tensor to update (in-place).
        grad (Tensor): Gradient tensor.
        exp_avg (Tensor): Exponential moving average of gradient.
        lr (float): Learning rate.
        beta1 (float): Coefficient used to form the update direction.
        beta2 (float): Coefficient for updating exp_avg.
        weight_decay (float): Decoupled weight decay.

    Returns:
        tuple: Updated (param, exp_avg)
    """
    update = exp_avg.mul(beta1).add(grad, alpha=1 - beta1)

    if weight_decay != 0:
        param.mul_(1 - lr * weight_decay)

    param.add_(torch.sign(update), alpha=-lr)

    exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

    return param, exp_avg

##################################################################################################################################################


import torch

def test_lion_step():
    results = {}

    # Test case 1: Basic step
    param1 = torch.randn(128, device='cuda')
    grad1 = torch.randn(128, device='cuda')
    exp_avg1 = torch.zeros(128, device='cuda')
    results["test_case_1"] = lion_step(param1.clone(), grad1, exp_avg1.clone(), lr=1e-3, beta1=0.9, beta2=0.99)

    # Test case 2: With weight decay
    param2 = torch.randn(256, device='cuda')
    grad2 = torch.randn(256, device='cuda')
    exp_avg2 = torch.zeros(256, device='cuda')
    results["test_case_2"] = lion_step(param2.clone(), grad2, exp_avg2.clone(), lr=1e-4, beta1=0.95, beta2=0.98, weight_decay=0.1)

    # Test case 3: Non-zero exp_avg
    param3 = torch.randn(64, device='cuda')
    grad3 = torch.randn(64, device='cuda')
    exp_avg3 = torch.randn(64, device='cuda') * 0.1
    results["test_case_3"] = lion_step(param3.clone(), grad3, exp_avg3.clone(), lr=5e-4, beta1=0.9, beta2=0.99)

    return results


test_results = test_lion_step()
