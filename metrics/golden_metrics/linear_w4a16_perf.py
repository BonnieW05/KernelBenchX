import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kernelbenchx.Quantization.linear_w4a16 import linear_w4a16
from performance_utils import Performance_Metrics
import torch
import torch.nn.functional as F

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=torch.float16, is_backward=False, **kwargs):
        super().__init__('linear_w4a16', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        configs = [
            (1, 4096, 4096),   # LLM decode, single token
            (8, 4096, 4096),   # small batch
            (32, 4096, 16384), # prefill
        ]
        for B, D_in, D_out in configs:
            input_tensor = torch.rand(B, D_in, dtype=self.dtype)
            weight = torch.rand(D_out, D_in, dtype=self.dtype)
            bias = torch.rand(D_out, dtype=self.dtype)
            self.input_tensors.append((input_tensor, weight, bias))

    def to_cuda(self, input_tensor):
        return (input_tensor[0].cuda(), input_tensor[1].cuda(), input_tensor[2].cuda())
    
    def call_op(self, input_tensor):
        return linear_w4a16(input_tensor[0], input_tensor[1], input_tensor[2])
    
    def get_gbps(self, input_tensor, runtime):
        inp, weight, bias = input_tensor
        B, D_in = inp.shape
        D_out = weight.shape[0]
        # Activation: fp16 (2 bytes/element)
        # Weight: int4 packed (0.5 bytes/element) — this is the key bandwidth saving
        # Bias: fp16, Output: fp16
        bytes_input = inp.numel() * 2          # fp16
        bytes_weight = weight.numel() * 0.5   # int4
        bytes_bias = bias.numel() * 2         # fp16
        bytes_output = B * D_out * 2          # fp16
        total_bytes = bytes_input + bytes_weight + bytes_bias + bytes_output
        return total_bytes / (runtime / 1000) / 1e9
    
    def get_tflops(self, input_tensor, runtime):
        inp, weight, _ = input_tensor
        B, D_in = inp.shape
        D_out = weight.shape[0]
        flops = 2 * B * D_in * D_out
        return flops / (runtime / 1000) / 1e12

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
