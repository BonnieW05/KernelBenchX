import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kernelbenchx.Math.signbit import signbit
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('signbit', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        dtype = self.dtype if self.dtype is not None else torch.float32
        for i in range(12, 28):
            size = 2 ** i
            if dtype.is_floating_point:
                input_tensor = torch.randn(size, dtype=dtype)
            else:
                input_tensor = torch.randint(-100, 100, (size,), dtype=dtype)
            self.input_tensors.append(input_tensor)

    def to_cuda(self, input_tensor):
        return input_tensor.cuda()
    
    def call_op(self, input_tensor):
        return signbit(input_tensor)
    
    def get_gbps(self, input_tensor, runtime):
        input_bytes = input_tensor.numel() * input_tensor.element_size()
        output_bytes = input_tensor.numel() * 1
        total_bytes = input_bytes + output_bytes
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        FLOPS = input_tensor.numel()
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS
    


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
