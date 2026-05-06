import torch
from typing import Tuple

def fused_cos_signbit(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the cosine of each element in the input tensor, followed by determining the sign bit 
    for each cosine result, indicating if it is positive or negative.
    
    Args:
        input (torch.Tensor): The input tensor for which the cosine and sign bit are computed.
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 
            - cos_result: The cosine of each element in the input tensor.
            - sign_bit: A boolean tensor indicating whether the cosine result is positive (False) or negative (True).
            
    Example:
        >>> a = torch.tensor([1.4309, 1.2706, -0.8562, 0.9796])
        >>> cos_result, sign_bit = cos_signbit(a)
        >>> cos_result
        tensor([ 0.1395,  0.2957,  0.6553,  0.5574])
        >>> sign_bit
        tensor([False, False, False, False])
    """
    cos_result = torch.cos(input)
    sign_bit = torch.signbit(cos_result)
    return (cos_result, sign_bit)

##################################################################################################################################################


import torch
from typing import Tuple
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_tensor

# def cos_signbit(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#     cos_result = torch.cos(input)
#     sign_bit = torch.signbit(cos_result)
#     return (cos_result, sign_bit)

def test_cos_signbit():
    results = {}

    # Test case 1: Positive values
    input_tensor_1 = torch.tensor([0.0, 1.0, 2.0], device='cuda')
    cos_result_1, sign_bit_1 = fused_cos_signbit(input_tensor_1)
    results["test_case_1"] = (cos_result_1.cpu(), sign_bit_1.cpu())

    # Test case 2: Negative values
    input_tensor_2 = torch.tensor([-1.0, -2.0, -3.0], device='cuda')
    cos_result_2, sign_bit_2 = fused_cos_signbit(input_tensor_2)
    results["test_case_2"] = (cos_result_2.cpu(), sign_bit_2.cpu())

    # Test case 3: Mixed values
    input_tensor_3 = torch.tensor([-1.0, 0.0, 1.0], device='cuda')
    cos_result_3, sign_bit_3 = fused_cos_signbit(input_tensor_3)
    results["test_case_3"] = (cos_result_3.cpu(), sign_bit_3.cpu())

    # Test case 4: Edge case with pi multiples
    input_tensor_4 = torch.tensor([torch.pi, -torch.pi, 2*torch.pi], device='cuda')
    cos_result_4, sign_bit_4 = fused_cos_signbit(input_tensor_4)
    results["test_case_4"] = (cos_result_4.cpu(), sign_bit_4.cpu())

    for mode in ("standard", "outlier"):
        outs = []
        for _ in range(3):
            x = rand_tensor((4096,), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            c, s = fused_cos_signbit(x)
            outs.append((c.cpu(), s.cpu()))
        results[f"test_random_{mode}"] = outs

    return results

test_results = test_cos_signbit()
