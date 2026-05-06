import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kernelbenchx.Fusion.fused_hstack_div import fused_hstack_div
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('fused_hstack_div', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(12, 28):
            num_elements = 2 ** i
            tensor1 = torch.rand(num_elements, dtype=self.dtype)
            tensor2 = torch.rand(num_elements, dtype=self.dtype)
            tensors = [tensor1, tensor2]
            divisor = 2.0
            self.input_tensors.append((tensors, divisor))
    
    def to_cuda(self, input_case):
        tensors, divisor = input_case
        cuda_tensors = [t.cuda() for t in tensors]
        if isinstance(divisor, torch.Tensor):
            cuda_divisor = divisor.cuda()
        else:
            cuda_divisor = divisor
        return (cuda_tensors, cuda_divisor)
        
    def call_op(self, input_case):
        tensors, divisor = input_case
        return fused_hstack_div(tensors, divisor)

    def get_gbps(self, input_case, runtime):
        tensors, divisor = input_case
        input_elements = sum(t.numel() for t in tensors)
        output_elements = torch.hstack(tensors).numel()
        element_size = tensors[0].element_size()
        total_bytes = (input_elements + output_elements * 3) * element_size
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_case, runtime):
        tensors, divisor = input_case
        output_elements = torch.hstack(tensors).numel()
        TFLOPS = output_elements / (runtime / 1000) / 1e12
        return TFLOPS



if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
