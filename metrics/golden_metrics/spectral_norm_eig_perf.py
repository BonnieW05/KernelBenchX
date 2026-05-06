import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kernelbenchx.Normalization.spectral_norm_eig import spectral_norm_eig
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('spectral_norm_eig', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 8):
            n = 2 ** i
            input_tensor = torch.rand(n, n, dtype=self.dtype or torch.float32)
            self.input_tensors.append(input_tensor)
    
    def to_cuda(self, input_tensor):
        return input_tensor.cuda()
        
    def call_op(self, input_tensor):
        return spectral_norm_eig(input_tensor)
    
    def get_gbps(self, input_tensor, runtime):
        n = input_tensor.shape[-1]
        batch_numel = input_tensor.numel() // (n * n)
        es = input_tensor.element_size()
        input_bytes = input_tensor.numel() * es
        output_bytes = batch_numel * es
        # eig materializes complex eigenvalues/vectors internally; approximate an effective
        # working set for a coarse bandwidth estimate (avoids near-zero I/O efficiency).
        workspace_bytes = batch_numel * n * n * es * 6
        total_bytes = input_bytes + output_bytes + workspace_bytes
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        n = input_tensor.shape[-1]
        batch_numel = input_tensor.numel() // (n * n)
        flops_per_matrix = 32 * n ** 3
        total_flops = batch_numel * flops_per_matrix
        TFLOPS = total_flops / (runtime / 1000) / 1e12
        return TFLOPS



if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
