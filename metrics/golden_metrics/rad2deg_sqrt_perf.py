import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kernelbenchx.Math.rad2deg_sqrt import rad2deg_sqrt
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('rad2deg_sqrt', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(12, 28):
            size = 2 ** i
            input_tensor = torch.rand(size, dtype=self.dtype).abs()
            self.input_tensors.append(input_tensor)
    
    def to_cuda(self, input_tensor):
        return input_tensor.cuda()
    
    def call_op(self, input_tensor):
        return rad2deg_sqrt(input_tensor)
    
    def get_gbps(self, input_tensor, runtime):
        num_elements = input_tensor.numel()
        element_size = input_tensor.element_size()
        total_bytes = num_elements * element_size * 4
        gbps = total_bytes / (runtime / 1000) / 1e9
        return gbps
    
    def get_tflops(self, input_tensor, runtime):
        flops = input_tensor.numel() * 2
        tflops = flops / (runtime / 1000) / 1e12
        return tflops
    


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
