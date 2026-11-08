# W4A16
Work In Progress 🚧 GEMM with fp16 activations and int4 weights: `C = A · W_dequant`, the weights being dequantized inside the kernel. In LLM decoding M (the batch size) is small, so the kernel is memory bound and reading 4-bit weights instead of fp16 ones should get close to a 4x speedup.


## Format
Symmetric quantization, GPTQ / Marlin convention, groups of 128 rows along K:

```
scales[group][n]  = 2 · max(|W[k][n]| for k in group) / 15
W_int4[k][n]      = clamp(round(W[k][n] / scales[k / 128][n]) + 8, 0, 15)
W_dequant[k][n]   = (W_int4[k][n] − 8) · scales[k / 128][n]
```

| Buffer | Shape | Type |
| --- | --- | --- |
| `A` | M × K | fp16 |
| `W_int4_packed` | (K / 8) × N | uint32, 8 values packed along K |
| `scales` | (K / 128) × N | fp16 |
| `C` | M × N | fp16, fp32 accumulation |

The quantization runs once on the host ([quantize.cpp](quantize.cpp)), it is the reference specification of the format.

## Implementations
- **w4a16_cublas.cu**: baseline, cuBLAS fp16 GEMM on the dequantized weights (`CUBLAS_COMPUTE_32F`). It reads 4x more weight bytes than a W4A16 kernel.
- **w4a16_naive.cu**: first implementation, work in progress.

## Measurements
- The cuBLAS result on `W_dequant` is the correctness reference.
- The L2 is flushed before every run (the int4 weights of an 8192 × 8192 matrix fit in the 40 MB L2 of an A100).
- Every implementation is timed next to the cuBLAS baseline on the same data, the median times give the speedup.
- The bench sweeps M from 1 to 256 with K = N = 8192 and exports `M,N,K,time_us,gflops,gbps,pct_peak_bw,cublas_time_us,speedup_vs_cublas`.

```sh
./build/libthom test w4a16 cublas 1 8192 8192 10
./build/libthom bench w4a16 cublas
```
