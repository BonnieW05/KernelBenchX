import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kernelbenchx.Quantization.attention_w8a8 import attention_w8a8
from performance_utils import Performance_Metrics
import torch
import torch.nn.functional as F

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=torch.float32, is_backward=False, **kwargs):
        super().__init__('attention_w8a8', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        # LLM-scale sequence lengths to test SRAM blocking and memory bottlenecks
        configs = [(2, 512, 64), (2, 1024, 64), (2, 2048, 64), (1, 4096, 64)]
        for B, S, D in configs:
            query = torch.rand(B, S, D, dtype=self.dtype)
            key = torch.rand(B, S, D, dtype=self.dtype)
            value = torch.rand(B, S, D, dtype=self.dtype)
            self.input_tensors.append((query, key, value))

    def to_cuda(self, input_tensor):
        return (input_tensor[0].cuda(), input_tensor[1].cuda(), input_tensor[2].cuda())
    
    def call_op(self, input_tensor):
        return attention_w8a8(input_tensor[0], input_tensor[1], input_tensor[2])
    
    def get_gbps(self, input_tensor, runtime):
        query, key, value = input_tensor
        B, S, D = query.shape
        # Dynamic quantization: reads fp32 inputs, writes fp32 output
        # Memory traffic = Q, K, V reads + output write (all fp32)
        total_bytes = (query.numel() + key.numel() + value.numel() + B * S * D) * query.element_size()
        return total_bytes / (runtime / 1000) / 1e9
    
    def get_tflops(self, input_tensor, runtime):
        query, key, value = input_tensor
        B, S, D = query.shape
        flops = 2 * B * S * S * D + 2 * B * S * S * D
        return flops / (runtime / 1000) / 1e12

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
