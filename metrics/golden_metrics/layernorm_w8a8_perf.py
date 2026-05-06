import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kernelbenchx.Quantization.layernorm_w8a8 import layernorm_w8a8
from performance_utils import Performance_Metrics
import torch
import torch.nn.functional as F

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=torch.float32, is_backward=False, **kwargs):
        super().__init__('layernorm_w8a8', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        configs = [(32, 2048), (64, 4096), (32, 8192)]
        for B, D in configs:
            input_tensor = torch.rand(B, D, dtype=self.dtype)
            weight = torch.rand(D, dtype=self.dtype)
            bias = torch.rand(D, dtype=self.dtype)
            self.input_tensors.append((input_tensor, weight, bias))

    def to_cuda(self, input_tensor):
        return (input_tensor[0].cuda(), input_tensor[1].cuda(), input_tensor[2].cuda())
    
    def call_op(self, input_tensor):
        inp, weight, bias = input_tensor
        return layernorm_w8a8(inp, (inp.shape[-1],), weight, bias, 1e-5)
    
    def get_gbps(self, input_tensor, runtime):
        inp, weight, bias = input_tensor
        # Dynamic quantization: input/output are fp32, weight/bias are fp32
        total_bytes = (inp.numel() + weight.numel() + bias.numel() + inp.numel()) * inp.element_size()
        return total_bytes / (runtime / 1000) / 1e9
    
    def get_tflops(self, input_tensor, runtime):
        return 0

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
