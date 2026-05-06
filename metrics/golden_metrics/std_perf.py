import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kernelbenchx.Reduce.std import std
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, dim=None, correction=1, keepdim=False, **kwargs):
        super().__init__('std', dtype=dtype, is_backward=is_backward, **kwargs)
        self.dim = dim
        self.correction = correction
        self.keepdim = keepdim
        self.output_sizes = []

    def get_input_tensors(self):
        self.input_tensors = []
        self.output_sizes = []
        for i in range(12, 28):
            size = 2 ** i
            input_tensor = torch.rand(size, dtype=self.dtype or torch.float32)
            self.input_tensors.append(input_tensor)

    def to_cuda(self, input_tensor):
        return input_tensor.cuda()
    
    def call_op(self, input_tensor):
        return std(input_tensor, 
                  dim=self.dim,
                  correction=self.correction,
                  keepdim=self.keepdim)
    
    def get_gbps(self, input_tensor, runtime):
        output_size = std(input_tensor, dim=self.dim, correction=self.correction, keepdim=self.keepdim).numel()
        element_size = input_tensor.element_size()
        total_bytes = (input_tensor.numel() + output_size) * element_size
        return total_bytes / (runtime / 1000) / 1e9
    
    def get_tflops(self, input_tensor, runtime):
        flops = 3 * input_tensor.numel()
        return flops / (runtime / 1000) / 1e12
    


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
