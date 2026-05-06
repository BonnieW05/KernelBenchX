import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(repo_root, 'data'))

from kernelbenchx.Optimizer.Lion import lion_step
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('Lion', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        dtype = self.dtype if self.dtype is not None else torch.get_default_dtype()
        element_size = torch.empty((), dtype=dtype).element_size()
        if torch.cuda.is_available():
            free_bytes, _ = torch.cuda.mem_get_info()
        else:
            free_bytes = 2 ** 30

        bytes_per_elem_alloc = 3 * element_size
        max_elements = int((free_bytes * 0.5) // max(bytes_per_elem_alloc, 1))
        for i in range(10, 29):
            size = 2 ** i
            if size > max_elements:
                break
            self.input_tensors.append(size)
        if len(self.input_tensors) == 0:
            self.input_tensors.append(2 ** 10)

    def to_cuda(self, input_tuple):
        size = input_tuple
        dtype = self.dtype if self.dtype is not None else torch.get_default_dtype()
        param = torch.randn(size, device='cuda', dtype=dtype)
        grad = torch.randn(size, device='cuda', dtype=dtype)
        exp_avg = torch.zeros(size, device='cuda', dtype=dtype)
        return (param, grad, exp_avg)
    
    def call_op(self, input_tuple):
        param, grad, exp_avg = input_tuple
        return lion_step(param, grad, exp_avg)
    
    def get_gbps(self, input_tuple, runtime):
        param, grad, exp_avg = input_tuple
        num_elements = param.numel()
        total_bytes = num_elements * 5 * param.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tuple, runtime):
        param, grad, exp_avg = input_tuple
        num_elements = param.numel()
        flops = num_elements * 6
        TFLOPS = flops / (runtime / 1000) / 1e12
        return TFLOPS


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
