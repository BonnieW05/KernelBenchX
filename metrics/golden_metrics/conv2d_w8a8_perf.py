import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kernelbenchx.Quantization.conv2d_w8a8 import conv2d_w8a8
from performance_utils import Performance_Metrics
import torch
import torch.nn.functional as F

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=torch.float32, is_backward=False, **kwargs):
        super().__init__('conv2d_w8a8', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        # ImageNet/ResNet scale: 224x224 with realistic channel counts
        configs = [(1, 3, 224, 224, 64, 7, 7), (2, 64, 56, 56, 128, 3, 3), (4, 128, 28, 28, 256, 3, 3)]
        for B, C_in, H, W, C_out, K_h, K_w in configs:
            input_tensor = torch.rand(B, C_in, H, W, dtype=self.dtype)
            weight = torch.rand(C_out, C_in, K_h, K_w, dtype=self.dtype)
            self.input_tensors.append((input_tensor, weight))

    def to_cuda(self, input_tensor):
        return (input_tensor[0].cuda(), input_tensor[1].cuda())
    
    def call_op(self, input_tensor):
        inp, weight = input_tensor
        return conv2d_w8a8(inp, weight, bias=None, stride=1, padding=1)
    
    def get_gbps(self, input_tensor, runtime):
        inp, weight = input_tensor
        B, C_in, H, W = inp.shape
        C_out, _, K_h, K_w = weight.shape
        # Output size with padding=1
        padding = 1  # Must match call_op (or parameterize and keep consistent).
        stride = 1
        H_out = (H + 2 * padding - K_h) // stride + 1
        W_out = (W + 2 * padding - K_w) // stride + 1
        # Dynamic quantization: reads fp32 inputs, writes fp32 output
        total_bytes = (inp.numel() + weight.numel() + B * C_out * H_out * W_out) * inp.element_size()
        return total_bytes / (runtime / 1000) / 1e9
    
    def get_tflops(self, input_tensor, runtime):
        inp, weight = input_tensor
        B, C_in, H, W = inp.shape
        C_out, _, K_h, K_w = weight.shape
        padding = 1
        stride = 1
        H_out = (H + 2 * padding - K_h) // stride + 1
        W_out = (W + 2 * padding - K_w) // stride + 1
        flops = 2 * B * C_out * (H_out) * (W_out) * C_in * K_h * K_w
        return flops / (runtime / 1000) / 1e12

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
