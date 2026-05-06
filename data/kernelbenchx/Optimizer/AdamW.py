import torch

def adamw_step(param, grad, exp_avg, exp_avg_sq, step, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
    """
    AdamW optimizer step (Adam with decoupled weight decay).
    
    Args:
        param (Tensor): Parameter tensor to update
        grad (Tensor): Gradient tensor
        exp_avg (Tensor): Exponential moving average of gradient
        exp_avg_sq (Tensor): Exponential moving average of squared gradient
        step (int): Current step number
        lr (float): Learning rate
        beta1 (float): Coefficient for first moment
        beta2 (float): Coefficient for second moment
        eps (float): Term added for numerical stability
        weight_decay (float): Weight decay coefficient
        
    Returns:
        tuple: Updated (param, exp_avg, exp_avg_sq)
    """
    # Decoupled weight decay
    param.mul_(1 - lr * weight_decay)
    
    # Update biased first moment estimate
    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
    
    # Update biased second raw moment estimate
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
    
    # Compute bias correction
    bias_correction1 = 1 - beta1 ** step
    bias_correction2 = 1 - beta2 ** step
    
    # Compute step size
    step_size = lr / bias_correction1
    bias_correction2_sqrt = (bias_correction2 ** 0.5)
    
    # Update parameters
    denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
    param.addcdiv_(exp_avg, denom, value=-step_size)
    
    return param, exp_avg, exp_avg_sq

##################################################################################################################################################


import torch

def test_adamw_step():
    results = {}

    # Test case 1: Basic AdamW step
    param1 = torch.randn(10, 5, device='cuda', requires_grad=False)
    grad1 = torch.randn(10, 5, device='cuda')
    exp_avg1 = torch.zeros(10, 5, device='cuda')
    exp_avg_sq1 = torch.zeros(10, 5, device='cuda')
    results["test_case_1"] = adamw_step(param1.clone(), grad1, exp_avg1.clone(), exp_avg_sq1.clone(), step=1)

    # Test case 2: With weight decay
    param2 = torch.randn(10, 5, device='cuda')
    grad2 = torch.randn(10, 5, device='cuda')
    exp_avg2 = torch.zeros(10, 5, device='cuda')
    exp_avg_sq2 = torch.zeros(10, 5, device='cuda')
    results["test_case_2"] = adamw_step(param2.clone(), grad2, exp_avg2.clone(), exp_avg_sq2.clone(), step=1, weight_decay=0.1)

    # Test case 3: Later step (for bias correction)
    param3 = torch.randn(10, 5, device='cuda')
    grad3 = torch.randn(10, 5, device='cuda')
    exp_avg3 = torch.randn(10, 5, device='cuda') * 0.1
    exp_avg_sq3 = torch.randn(10, 5, device='cuda').abs() * 0.01
    results["test_case_3"] = adamw_step(param3.clone(), grad3, exp_avg3.clone(), exp_avg_sq3.clone(), step=100)

    return results

test_results = test_adamw_step()
