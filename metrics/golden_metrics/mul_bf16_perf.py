import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from kernelbenchx.Math.mul_bf16 import mul_bf16
from performance_utils import Performance_Metrics
import torch

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=torch.bfloat16, is_backward=False, **kwargs):
        super().__init__('mul_bf16', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(8, 14):
            size = 2 ** i
            self.input_tensors.append((torch.rand(size, size, dtype=self.dtype), torch.rand(size, size, dtype=self.dtype)))

    def to_cuda(self, input_tensor):
        return (input_tensor[0].cuda(), input_tensor[1].cuda())
    
    def call_op(self, input_tensor):
        return mul_bf16(input_tensor[0], input_tensor[1])
    
    def get_gbps(self, input_tensor, runtime):
        total_bytes = (input_tensor[0].numel() + input_tensor[1].numel() + input_tensor[0].numel()) * input_tensor[0].element_size()
        return total_bytes / (runtime / 1000) / 1e9
    
    def get_tflops(self, input_tensor, runtime):
        flops = input_tensor[0].numel()
        return flops / (runtime / 1000) / 1e12

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
