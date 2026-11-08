#include "w4a16.h"
#include <cmath>

/* Offline quantization of the weights, done once on the host before any benchmark.
This file is the reference specification of the format: kernels must reproduce dequantize_w4 exactly. */


void quantize_w4(const __half * W, uint8_t * W_int4, __half * scales, int K, int N) {
    const int nbGroups = K / GROUP_SIZE;
    std::vector<float> maxAbs(N);

    for (int group = 0; group < nbGroups; group++) {
        // 1 - max |W| of each column of the group
        std::fill(maxAbs.begin(), maxAbs.end(), 0.0f);
        for (int k = group * GROUP_SIZE; k < (group + 1) * GROUP_SIZE; k++) {
            for (int n = 0; n < N; n++) {
                maxAbs[n] = std::max(maxAbs[n], std::fabs(__half2float(W[k * N + n])));
            }
        }

        // 2 - one fp16 scale per column. An all-zero group gets scale 1 so we never divide by 0 (its values all quantize to ZERO_POINT)
        for (int n = 0; n < N; n++) {
            float scale = 2.0f * maxAbs[n] / 15.0f;
            scales[group * N + n] = __float2half(scale > 0.0f ? scale : 1.0f);
        }

        // 3 - quantize with the fp16-rounded scale (the one the kernel will see), so that dequantization is consistent
        for (int k = group * GROUP_SIZE; k < (group + 1) * GROUP_SIZE; k++) {
            for (int n = 0; n < N; n++) {
                float scale = __half2float(scales[group * N + n]);
                // nearbyint uses the default rounding mode: round half to even, like numpy / torch
                int q = (int)std::nearbyint(__half2float(W[k * N + n]) / scale) + ZERO_POINT;
                W_int4[k * N + n] = (uint8_t)std::min(std::max(q, 0), 15);
            }
        }
    }
}


/* GPTQ layout: W_int4_packed[k / 8][n] holds rows 8*(k/8) ... 8*(k/8)+7 of column n,
row k is stored in bits 4*(k%8) ... 4*(k%8)+3 */
void pack_w4(const uint8_t * W_int4, uint32_t * W_int4_packed, int K, int N) {
    for (int packedRow = 0; packedRow < K / INT4_PER_UINT32; packedRow++) {
        for (int n = 0; n < N; n++) {
            uint32_t packed = 0;
            for (int i = 0; i < INT4_PER_UINT32; i++) {
                uint32_t q = W_int4[(packedRow * INT4_PER_UINT32 + i) * N + n];
                packed |= q << (4 * i);
            }
            W_int4_packed[packedRow * N + n] = packed;
        }
    }
}


/* The fp32 product of two fp16 values is exact (11 + 11 significand bits < 24), so rounding it once to fp16
gives the same bits as a fp16 multiply on the GPU (__hmul / __hmul2) */
void dequantize_w4(const uint8_t * W_int4, const __half * scales, __half * W_dequant, int K, int N) {
    for (int k = 0; k < K; k++) {
        for (int n = 0; n < N; n++) {
            float scale = __half2float(scales[(k / GROUP_SIZE) * N + n]);
            W_dequant[k * N + n] = __float2half((float)((int)W_int4[k * N + n] - ZERO_POINT) * scale);
        }
    }
}
