import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kernelbenchx.Quantization.matmul_w8a8 import matmul_w8a8
from performance_utils import Performance_Metrics
import torch

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=torch.float32, is_backward=False, **kwargs):
        super().__init__('matmul_w8a8', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        sizes = [(128, 256, 512), (256, 512, 1024), (512, 1024, 2048), (1024, 2048, 4096)]
        for M, K, N in sizes:
            input_tensor = (torch.rand(M, K, dtype=self.dtype), torch.rand(K, N, dtype=self.dtype))
            self.input_tensors.append(input_tensor)

    def to_cuda(self, input_tensor):
        return (input_tensor[0].cuda(), input_tensor[1].cuda())
    
    def call_op(self, input_tensor):
        return matmul_w8a8(input_tensor[0], input_tensor[1])
    
    def get_gbps(self, input_tensor, runtime):
        input_a, input_b = input_tensor
        M, K = input_a.shape
        N = input_b.shape[1]
        # Dynamic quantization: reads fp32 inputs, writes fp32 output
        total_bytes = (input_a.numel() + input_b.numel() + M * N) * input_a.element_size()
        return total_bytes / (runtime / 1000) / 1e9
    
    def get_tflops(self, input_tensor, runtime):
        input_a, input_b = input_tensor
        M, K = input_a.shape
        N = input_b.shape[1]
        flops = 2 * M * K * N
        return flops / (runtime / 1000) / 1e12

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
