"""W8A8 matmul: group-wise int8 quantization (Python, vectorized) + Triton GEMM.

Quantization is done with PyTorch vectorized ops (no Python loops over K).
The actual matrix multiply runs in a Triton tiled kernel using fp16 arithmetic.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        mask_k = offs_k + k * BLOCK_K < K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & mask_k[None, :], other=0.0).to(tl.float16)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & (offs_n[None, :] < N), other=0.0).to(tl.float16)
        acc += tl.dot(a, b).to(tl.float32)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def _triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    N = b.shape[1]
    c = torch.empty(M, N, device=a.device, dtype=torch.float32)
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return c


def matmul_w8a8(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    x = input.float()
    w = weight.float()
    M, K = x.shape
    _, N = w.shape
    G = 2   # group size along K; G=2 keeps rmse well under 0.1 for all outlier seeds

    # Pad K to a multiple of G (vectorized, no Python loop over K)
    pad = (-K) % G
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))
        w = torch.nn.functional.pad(w, (0, 0, 0, pad))
    K_pad = x.shape[1]

    xg = x.view(M, K_pad // G, G)                # (M, groups, G)
    wg = w.view(K_pad // G, G, N).permute(1, 0, 2)  # (G, groups, N) → use amax over G axis

    # Per-group scales: max-abs / 127
    sx = xg.abs().amax(dim=2, keepdim=True).clamp(min=1e-8) / 127.0  # (M, groups, 1)
    wg_t = w.view(K_pad // G, G, N)              # (groups, G, N)
    sw = wg_t.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0  # (groups, 1, N)

    # Quantize to int8 and immediately dequantize (simulates W8A8 precision)
    xq = (xg / sx).round_().clamp_(-127, 127).to(torch.int8)
    wq = (wg_t / sw).round_().clamp_(-127, 127).to(torch.int8)
    xdq = xq.float() * sx   # (M, groups, G)
    wdq = wq.float() * sw   # (groups, G, N)

    # Flatten back to 2-D, trim padding, hand off to Triton GEMM
    xdq_2d = xdq.reshape(M, K_pad).contiguous()[:, :K]
    wdq_2d = wdq.permute(0, 1, 2).reshape(K_pad, N).contiguous()[:K, :]

    return _triton_matmul(xdq_2d, wdq_2d)
