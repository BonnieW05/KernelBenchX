import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kernelbenchx.Index.permute_copy import permute_copy
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('permute_copy', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(3, 8):
            print(i)
            dim_size = 2 ** i
            shape = (dim_size, dim_size, dim_size, dim_size)
            input_tensor = torch.rand(shape, dtype=self.dtype or torch.float32)
            self.input_tensors.append((input_tensor,))
    
    def to_cuda(self, input_tuple):
        (input_tensor,) = input_tuple
        return (input_tensor.cuda(),)
        
    def call_op(self, input_tuple):
        (input_tensor,) = input_tuple
        return permute_copy(input_tensor, dims=[3, 2, 1, 0])
    
    def get_gbps(self, input_tuple, runtime):
        (input_tensor,) = input_tuple
        total_bytes = 2 * input_tensor.numel() * input_tensor.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tuple, runtime):
        (input_tensor,) = input_tuple
        FLOPS = input_tensor.numel()
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS
        
if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
