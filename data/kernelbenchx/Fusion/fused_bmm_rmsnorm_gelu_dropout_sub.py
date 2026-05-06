import torch
import torch.nn.functional as F

def fused_bmm_rmsnorm_gelu_dropout_sub(input1, input2, other, normalized_shape, dropout_p=0.5, training=True, approximate='none', eps=1e-05, *, out=None):
    z1 = torch.bmm(input1, input2)
    rms_norm = F.rms_norm(z1, normalized_shape=(normalized_shape,), eps=eps)
    gelu_out = F.gelu(rms_norm, approximate=approximate)
    output = F.dropout(gelu_out, p=dropout_p, training=training)
    if out is not None:
        out.copy_(output)
        return out
    return output

##################################################################################################################################################


import torch
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))
from data_utils import rand_tensor

def test_fused_bmm_rmsnorm_gelu_dropout_sub():
    results = {}

    # Test case 1: Basic test with default parameters
    input1 = torch.randn(2, 3, 4, device='cuda')
    input2 = torch.randn(2, 4, 5, device='cuda')
    other = torch.randn(2, 3, 5, device='cuda')
    normalized_shape = 5
    results["test_case_1"] = fused_bmm_rmsnorm_gelu_dropout_sub(input1, input2, other, normalized_shape)

    # Test case 2: Test with different dropout probability
    dropout_p = 0.3
    results["test_case_2"] = fused_bmm_rmsnorm_gelu_dropout_sub(input1, input2, other, normalized_shape, dropout_p=dropout_p)

    # Test case 3: Test with training set to False
    training = False
    results["test_case_3"] = fused_bmm_rmsnorm_gelu_dropout_sub(input1, input2, other, normalized_shape, training=training)

    # Test case 4: Test with approximate GELU
    approximate = 'tanh'
    results["test_case_4"] = fused_bmm_rmsnorm_gelu_dropout_sub(input1, input2, other, normalized_shape, approximate=approximate)

    for mode in ("standard", "outlier"):
        outs = []
        for training in (False, True):
            x1 = rand_tensor((4, 16, 32), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            x2 = rand_tensor((4, 32, 64), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            o = rand_tensor((4, 16, 64), dtype=torch.float32, mode=mode, outlier_prob=0.001, outlier_scale=10.0)
            outs.append(
                fused_bmm_rmsnorm_gelu_dropout_sub(
                    x1,
                    x2,
                    o,
                    normalized_shape=64,
                    dropout_p=0.1,
                    training=training,
                    approximate="tanh",
                )
            )
        results[f"test_random_{mode}"] = outs

    return results

test_results = test_fused_bmm_rmsnorm_gelu_dropout_sub()
