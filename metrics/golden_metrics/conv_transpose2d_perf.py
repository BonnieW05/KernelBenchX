import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(repo_root, 'data'))

from kernelbenchx.Convolution.conv_transpose2d import conv_transpose2d
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('conv_transpose2d', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        configs = []
        for i in range(4, 10):
            N = 2
            C_in = 2 ** (i - 1)
            C_out = 2 ** i
            H = 16
            W = 16
            kH = 3
            kW = 3
            stride = 2
            padding = 1
            output_padding = 1
            dilation = 1
            groups = 1
            configs.append(((N, C_in, H, W), C_out, (kH, kW), stride, padding, output_padding, dilation, groups))

        for input_shape, out_channels, kernel_size, stride, padding, output_padding, dilation, groups in configs:
            input_tensor = torch.randn(*input_shape, dtype=self.dtype or torch.float32)
            in_channels = input_shape[1]
            kH, kW = kernel_size
            weight = torch.randn(in_channels, out_channels // groups, kH, kW, dtype=self.dtype or torch.float32)
            bias = torch.randn(out_channels, dtype=self.dtype or torch.float32)
            self.input_tensors.append((input_tensor, weight, bias, stride, padding, output_padding, groups, dilation))

    def to_cuda(self, input_tuple):
        input_tensor, weight, bias, stride, padding, output_padding, groups, dilation = input_tuple
        return (input_tensor.cuda(), weight.cuda(), bias.cuda(), stride, padding, output_padding, groups, dilation)

    def call_op(self, input_tuple):
        input_tensor, weight, bias, stride, padding, output_padding, groups, dilation = input_tuple
        return conv_transpose2d(input_tensor, weight, bias=bias, stride=stride, padding=padding, output_padding=output_padding, groups=groups, dilation=dilation)

    def get_gbps(self, input_tuple, runtime):
        input_tensor, weight, bias, stride, padding, output_padding, groups, dilation = input_tuple
        N, C_in, H_in, W_in = input_tensor.shape
        _, C_out_per_group, kH, kW = weight.shape
        C_out = C_out_per_group * groups

        if isinstance(stride, (tuple, list)):
            sH, sW = stride
        else:
            sH = sW = stride
        if isinstance(padding, (tuple, list)):
            pH, pW = padding
        else:
            pH = pW = padding
        if isinstance(output_padding, (tuple, list)):
            opH, opW = output_padding
        else:
            opH = opW = output_padding
        if isinstance(dilation, (tuple, list)):
            dH, dW = dilation
        else:
            dH = dW = dilation

        H_out = (H_in - 1) * sH - 2 * pH + dH * (kH - 1) + opH + 1
        W_out = (W_in - 1) * sW - 2 * pW + dW * (kW - 1) + opW + 1
        output_numel = N * C_out * H_out * W_out

        element_size = input_tensor.element_size()
        input_bytes = input_tensor.numel() * element_size
        weight_bytes = weight.numel() * element_size
        bias_bytes = bias.numel() * element_size if bias is not None else 0
        output_bytes = output_numel * element_size
        total_bytes = input_bytes + weight_bytes + bias_bytes + output_bytes

        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS

    def get_tflops(self, input_tuple, runtime):
        input_tensor, weight, bias, stride, padding, output_padding, groups, dilation = input_tuple
        N, C_in, H_in, W_in = input_tensor.shape
        _, C_out_per_group, kH, kW = weight.shape
        C_out = C_out_per_group * groups

        if isinstance(stride, (tuple, list)):
            sH, sW = stride
        else:
            sH = sW = stride
        if isinstance(padding, (tuple, list)):
            pH, pW = padding
        else:
            pH = pW = padding
        if isinstance(output_padding, (tuple, list)):
            opH, opW = output_padding
        else:
            opH = opW = output_padding
        if isinstance(dilation, (tuple, list)):
            dH, dW = dilation
        else:
            dH = dW = dilation

        flops_per_input = 2 * kH * kW * (C_out // groups)
        total_flops = N * C_in * H_in * W_in * flops_per_input

        if bias is not None:
            H_out = (H_in - 1) * sH - 2 * pH + dH * (kH - 1) + opH + 1
            W_out = (W_in - 1) * sW - 2 * pW + dW * (kW - 1) + opW + 1
            total_flops += N * C_out * H_out * W_out

        TFLOPS = total_flops / (runtime / 1000) / 1e12
        return TFLOPS


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
