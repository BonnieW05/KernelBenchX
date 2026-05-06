import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(repo_root, 'data'))

from kernelbenchx.Pooling.avg_pool1d import avg_pool1d
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('avg_pool1d', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(10, 16):
            size = 2 ** i
            input_tensor = torch.randn(2, 64, size, dtype=self.dtype)
            self.input_tensors.append(input_tensor)

    def to_cuda(self, input_tensor):
        return input_tensor.cuda()
    
    def call_op(self, input_tensor):
        return avg_pool1d(input_tensor, kernel_size=2)
    
    def get_gbps(self, input_tensor, runtime):
        batch, channels, length_in = input_tensor.shape
        element_size = input_tensor.element_size()
        input_bytes = batch * channels * length_in * element_size
        
        # L_out = floor((L_in + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1)
        length_out = (length_in - 2) // 2 + 1 
        output_bytes = batch * channels * length_out * element_size
        
        total_bytes = input_bytes + output_bytes
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        batch, channels, length = input_tensor.shape
        flops = batch * channels * (length // 2) * 2
        TFLOPS = flops / (runtime / 1000) / 1e12
        return TFLOPS


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
