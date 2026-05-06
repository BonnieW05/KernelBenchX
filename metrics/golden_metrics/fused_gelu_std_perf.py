import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kernelbenchx.Fusion.fused_gelu_std import fused_gelu_std
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('fused_gelu_std', dtype=dtype, is_backward=is_backward, **kwargs)
        self.output_sizes = []

    def get_input_tensors(self):
        self.input_tensors = []
        self.output_sizes = []
        for i in range(12, 28):
            size = 2 ** i
            input_tensor = torch.rand(size, dtype=torch.float32)
            output_tensor = self.call_op(input_tensor)
            self.input_tensors.append(input_tensor)
            self.output_sizes.append(output_tensor.numel())

    def to_cuda(self, input_tensor):
        return input_tensor.cuda()
    
    def call_op(self, input_tensor):
        return fused_gelu_std(input_tensor)
    
    def get_gbps(self, input_tensor, runtime):
        shape = input_tensor.shape
        index = None
        for i, tensor in enumerate(self.input_tensors):
            if tensor.shape == shape:
                index = i
                break
        if index is None:
            raise ValueError("Input tensor shape not found in precomputed list.")
        output_size = self.output_sizes[index]
        total_bytes = (input_tensor.numel() + output_size) * input_tensor.element_size() * 2
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        flops_per_element = 11
        total_flops = input_tensor.numel() * flops_per_element
        TFLOPS = total_flops / (runtime / 1000) / 1e12
        return TFLOPS
    


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
