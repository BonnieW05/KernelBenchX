import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from kernelbenchx.Activation.sigmoid_fp16 import sigmoid_fp16
from performance_utils import Performance_Metrics
import torch

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=torch.float16, is_backward=False, **kwargs):
        super().__init__('sigmoid_fp16', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(12, 28):
            size = 2 ** i
            self.input_tensors.append(torch.rand(size, dtype=self.dtype))

    def to_cuda(self, input_tensor):
        return input_tensor.cuda()
    
    def call_op(self, input_tensor):
        return sigmoid_fp16(input_tensor)
    
    def get_gbps(self, input_tensor, runtime):
        total_bytes = input_tensor.numel() * input_tensor.element_size() * 2
        return total_bytes / (runtime / 1000) / 1e9
    
    def get_tflops(self, input_tensor, runtime):
        flops = input_tensor.numel() * 5
        return flops / (runtime / 1000) / 1e12

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
