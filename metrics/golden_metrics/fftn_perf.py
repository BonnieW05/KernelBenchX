import sys
import os
import json
import math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kernelbenchx.Math.fftn import fftn
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('fftn', dtype=dtype, is_backward=is_backward, **kwargs)
        self.s = kwargs.get('s', None)
        self.dim = kwargs.get('dim', None)
        self.norm = kwargs.get('norm', None)

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(12, 24):
            size = 2 ** i
            input_tensor = torch.randn(size, dtype=torch.complex64)
            self.input_tensors.append(input_tensor)

    def to_cuda(self, input_tensor):
        return input_tensor.cuda()

    def call_op(self, input_tensor):
        return fftn(input_tensor, s=self.s, dim=self.dim, norm=self.norm)

    def get_gbps(self, input_tensor, runtime):
        input_element_size = input_tensor.element_size()
        num_elements = input_tensor.numel()
        
        if input_tensor.is_complex():
            output_element_size = input_element_size
        else:
            output_element_size = torch.tensor([], dtype=torch.complex64).element_size()
        
        total_bytes = num_elements * (input_element_size + output_element_size)
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS

    def get_tflops(self, input_tensor, runtime):
        N = input_tensor.numel()
        if N == 0:
            return 0.0
        log2_N = math.log2(N)
        flops = 5 * N * log2_N
        TFLOPS = flops / (runtime / 1000) / 1e12
        return TFLOPS
    


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
