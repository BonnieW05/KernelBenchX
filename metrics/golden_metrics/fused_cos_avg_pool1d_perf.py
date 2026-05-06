import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kernelbenchx.Fusion.fused_cos_avg_pool1d import fused_cos_avg_pool1d
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('fused_cos_avg_pool1d', dtype=dtype, is_backward=is_backward, **kwargs)
        self.kernel_size = 3

    def get_input_tensors(self):
        self.input_tensors = []
        k = self.kernel_size
        stride = k  # Match F.avg_pool1d default; avoid extra preprocessing passes (can be slow / hit timeouts).
        for i in range(12, 24):
            W = 128 * i
            input_shape = (32, 32, W)  # (minibatch, in_channels, iW)
            input_tensor = torch.rand(*input_shape, dtype=self.dtype or torch.float32)
            L_out = (W + 2 * 0 - (k - 1) - 1) // stride + 1
            out_shape = (32, 32, L_out)
            self.input_tensors.append((input_tensor, out_shape))

    def to_cuda(self, input_tensor):
        input_tensor_, _ = input_tensor
        return (input_tensor_.cuda(), _)
    
    def call_op(self, input_tensor):
        input_tensor_, _ = input_tensor
        return fused_cos_avg_pool1d(input_tensor_, kernel_size=self.kernel_size)
    
    def get_gbps(self, input_tensor, runtime):
        # idx = self.input_tensors.index(input_tensor)
        # output_shape = self.output_shapes[idx]
        input_tensor_, output_shape = input_tensor
        output_numel = torch.Size(output_shape).numel()
        
        element_size = input_tensor_.element_size()
        total_bytes = (2 * input_tensor_.numel() + output_numel) * element_size
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        # idx = self.input_tensors.index(input_tensor)
        # output_shape = self.output_shapes[idx]
        input_tensor_, output_shape = input_tensor
        output_numel = torch.Size(output_shape).numel()
        
        flops_cos = input_tensor_.numel()
        flops_pool = output_numel * self.kernel_size
        total_flops = flops_cos + flops_pool
        
        TFLOPS = total_flops / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
