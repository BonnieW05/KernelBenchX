import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kernelbenchx.Reduce.min_gelu import min_gelu
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, approximate='none', **kwargs):
        super().__init__('min_gelu', dtype=dtype, is_backward=is_backward, **kwargs)
        self.approximate = approximate

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(12, 28):
            size = 2 ** i
            input_tensor = torch.rand(size, dtype=self.dtype)
            self.input_tensors.append(input_tensor)

    def to_cuda(self, input_tensor):
        return input_tensor.cuda()
    
    def call_op(self, input_tensor):
        return min_gelu(input_tensor, dim=0, approximate=self.approximate)
    
    def get_gbps(self, input_tensor, runtime):
        element_size = input_tensor.element_size()
        total_bytes = (3 * input_tensor.numel() + 1) * element_size
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        if self.approximate == 'none':
            gelu_flops_per_element = 8
        else:
            gelu_flops_per_element = 12
            
        num_elements = input_tensor.numel()
        total_flops = num_elements * gelu_flops_per_element + (num_elements - 1)
        TFLOPS = total_flops / (runtime / 1000) / 1e12
        return TFLOPS



if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
