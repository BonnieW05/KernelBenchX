import torch

def rmsprop_step(param, grad, square_avg, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0.0):
    """RMSProp optimizer step.

    Args:
        param (Tensor): Parameter tensor to update (in-place).
        grad (Tensor): Gradient tensor.
        square_avg (Tensor): Exponential moving average of squared gradients.
        lr (float): Learning rate.
        alpha (float): Smoothing constant for running average.
        eps (float): Term added for numerical stability.
        weight_decay (float): L2 penalty.

    Returns:
        tuple: Updated (param, square_avg)
    """
    if weight_decay != 0:
        grad = grad.add(param, alpha=weight_decay)

    square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)
    avg = square_avg.sqrt().add(eps)
    param.addcdiv_(grad, avg, value=-lr)

    return param, square_avg

##################################################################################################################################################


import torch

def test_rmsprop_step():
    results = {}

    # Test case 1: Basic step
    param1 = torch.randn(128, device='cuda')
    grad1 = torch.randn(128, device='cuda')
    square_avg1 = torch.zeros(128, device='cuda')
    results["test_case_1"] = rmsprop_step(param1.clone(), grad1, square_avg1.clone(), lr=1e-2, alpha=0.99, eps=1e-8)

    # Test case 2: With weight decay
    param2 = torch.randn(256, device='cuda')
    grad2 = torch.randn(256, device='cuda')
    square_avg2 = torch.zeros(256, device='cuda')
    results["test_case_2"] = rmsprop_step(param2.clone(), grad2, square_avg2.clone(), lr=1e-3, alpha=0.95, eps=1e-6, weight_decay=0.1)

    # Test case 3: Non-zero running average
    param3 = torch.randn(64, device='cuda')
    grad3 = torch.randn(64, device='cuda')
    square_avg3 = torch.rand(64, device='cuda') * 0.01
    results["test_case_3"] = rmsprop_step(param3.clone(), grad3, square_avg3.clone(), lr=5e-3, alpha=0.9, eps=1e-8)

    return results


test_results = test_rmsprop_step()
