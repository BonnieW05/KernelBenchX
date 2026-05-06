import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kernelbenchx.Quantization.bmm_w8a8 import bmm_w8a8
from performance_utils import Performance_Metrics
import torch

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=torch.float32, is_backward=False, **kwargs):
        super().__init__('bmm_w8a8', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        configs = [(2, 64, 128, 256), (4, 128, 256, 512), (8, 256, 512, 1024), (16, 512, 1024, 2048)]
        for B, M, K, N in configs:
            input_a = torch.rand(B, M, K, dtype=self.dtype)
            input_b = torch.rand(B, K, N, dtype=self.dtype)
            self.input_tensors.append((input_a, input_b))

    def to_cuda(self, input_tensor):
        return (input_tensor[0].cuda(), input_tensor[1].cuda())
    
    def call_op(self, input_tensor):
        return bmm_w8a8(input_tensor[0], input_tensor[1])
    
    def get_gbps(self, input_tensor, runtime):
        input_a, input_b = input_tensor
        B, M, K = input_a.shape
        N = input_b.shape[2]
        # Dynamic quantization: reads fp32 inputs, writes fp32 output
        total_bytes = (input_a.numel() + input_b.numel() + B * M * N) * input_a.element_size()
        return total_bytes / (runtime / 1000) / 1e9
    
    def get_tflops(self, input_tensor, runtime):
        input_a, input_b = input_tensor
        B, M, K = input_a.shape
        N = input_b.shape[2]
        flops = 2 * B * M * K * N
        return flops / (runtime / 1000) / 1e12

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
