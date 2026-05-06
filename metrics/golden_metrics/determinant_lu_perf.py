import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kernelbenchx.LinearAlgebra.determinant_lu import determinant_lu
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('determinant_lu', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        for exp in range(4, 11):
            n = 2 ** exp
            input_tensor = torch.randn(n, n, dtype=self.dtype or torch.float32)
            self.input_tensors.append(input_tensor)

    def to_cuda(self, input_tensor):
        return input_tensor.cuda()

    def call_op(self, input_tensor):
        return determinant_lu(input_tensor)

    def get_gbps(self, input_tensor, runtime):
        input_numel = input_tensor.numel()
        output_numel = 1
        for dim in input_tensor.shape[:-2]:
            output_numel *= dim
        total_bytes = (input_numel + output_numel) * input_tensor.element_size() * 2 * 6
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS

    def get_tflops(self, input_tensor, runtime):
        n = input_tensor.size(-1)
        batch_dims = input_tensor.shape[:-2]
        batch_size = 1
        for dim in batch_dims:
            batch_size *= dim
        flops_per_matrix = (2/3) * (n ** 3)
        total_flops = flops_per_matrix * batch_size
        TFLOPS = total_flops / (runtime / 1000) / 1e12
        return TFLOPS   
    


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
