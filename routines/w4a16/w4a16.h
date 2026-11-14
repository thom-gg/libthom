#pragma once

#include "../routines.h"
#include "cuda_fp16.h"
#include <cstdint>

/* W4A16 GEMM: C = A * W_dequant
    A:              M x K       fp16, row major (activations)
    W_int4_packed:  (K/8) x N   uint32, 8 int4 values packed along K (GPTQ layout)
    scales:         (K/GROUP_SIZE) x N   fp16, one scale per group of GROUP_SIZE rows and per column
    C:              M x N       fp16, row major

    Symmetric quantization (GPTQ / Marlin convention), zero point is fixed and never stored:
        scales[group][n]  = 2 * max(|W[k][n]| for k in group) / 15
        W_int4[k][n]      = clamp(round(W[k][n] / scales[k / GROUP_SIZE][n]) + ZERO_POINT, 0, 15)
        W_dequant[k][n]   = (W_int4[k][n] - ZERO_POINT) * scales[k / GROUP_SIZE][n]
*/
#define GROUP_SIZE 128
#define ZERO_POINT 8
#define INT4_PER_UINT32 8

/* Every buffer of the problem, on the GPU. An implementation only reads the ones matching its weight format */
struct W4A16Problem {
    const __half * A;
    const uint32_t * W_int4_packed;
    const __half * scales;
    const __half * W_dequant;   // K x N fp16, only used by fp16 baselines (cuBLAS)
    __half * C;
    size_t M;
    size_t N;
    size_t K;
};

using W4A16Function = float (*)(const W4A16Problem& problem);

/* Which weights an implementation reads, to count the bytes it moves */
enum class WeightFormat {
    FP16,
    INT4
};

struct W4A16Impl {
    W4A16Function function;
    WeightFormat weights;
};


/* Host-side quantization (quantize.cpp). W, W_int4 and W_dequant are K x N row major */
void quantize_w4(const __half * W, uint8_t * W_int4, __half * scales, int K, int N);
void pack_w4(const uint8_t * W_int4, uint32_t * W_int4_packed, int K, int N);
void dequantize_w4(const uint8_t * W_int4, const __half * scales, __half * W_dequant, int K, int N);


#if LIB_HAS(W4A16_CUBLAS)
float w4a16_cublas(const W4A16Problem& problem);
#endif

#if LIB_HAS(W4A16_NAIVE)
float w4a16_naive(const W4A16Problem& problem);
#endif

#if LIB_HAS(W4A16_PIPELINING)
float w4a16_pipelining(const W4A16Problem& problem);
#endif