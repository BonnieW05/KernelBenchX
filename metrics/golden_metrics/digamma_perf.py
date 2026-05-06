import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kernelbenchx.Math.digamma import digamma
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('digamma', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(12, 28):  # Sizes from 2^12 to 2^27
            size = 2 ** i
            # Ensure positive values to avoid domain errors in digamma
            input_tensor = torch.rand(size, dtype=torch.float32) + 1.0  # Shift to [1.0, 2.0)
            self.input_tensors.append(input_tensor)
    
    def to_cuda(self, input_tensor):
        return input_tensor.cuda()
        
    def call_op(self, input_tensor):
        return digamma(input_tensor)

    def get_gbps(self, input_tensor, runtime):
        # Total bytes = input + output (both have same shape/dtype)
        total_bytes = input_tensor.numel() * input_tensor.element_size() * 2
        GBPS = total_bytes / (runtime / 1000) / 1e9  # Convert ms to seconds
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        # Simplified assumption: 1 FLOP per element (actual FLOPs may vary)
        FLOPS = input_tensor.numel()  
        TFLOPS = FLOPS / (runtime / 1000) / 1e12  # Convert ms to seconds
        return TFLOPS



if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
