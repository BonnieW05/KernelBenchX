import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from kernelbenchx.MatrixMultiply.matmul_bf16 import matmul_bf16
from performance_utils import Performance_Metrics
import torch

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=torch.bfloat16, is_backward=False, **kwargs):
        super().__init__('matmul_bf16', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        # Matrix sizes aligned with the reference measurement set.
        sizes = [256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 
                 2048, 2176, 2304, 2432, 2560, 2688, 2816, 2944, 3072, 3200, 3328, 3456, 3584, 
                 3712, 3840, 3968, 4096]
        for n in sizes:
            a = torch.randn(n, n, dtype=self.dtype)
            b = torch.randn(n, n, dtype=self.dtype)
            self.input_tensors.append((a, b))

    def to_cuda(self, input_tensor):
        return (input_tensor[0].cuda(), input_tensor[1].cuda())
    
    def call_op(self, input_tensor):
        return matmul_bf16(input_tensor[0], input_tensor[1])
    
    def get_gbps(self, input_tensor, runtime):
        a, b = input_tensor
        total_bytes = (a.numel() + b.numel()) * a.element_size()
        return total_bytes / (runtime / 1000) / 1e9
    
    def get_tflops(self, input_tensor, runtime):
        a, b = input_tensor
        M, K = a.shape
        N = b.shape[1]
        flops = 2 * M * K * N
        return flops / (runtime / 1000) / 1e12

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
