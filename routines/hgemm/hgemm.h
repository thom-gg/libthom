#pragma once

#include "../routines.h"
#include "cuda_fp16.h"

using HgemmFunction = float (*)(const __half* A, const __half* B, __half* C,
                                size_t M, size_t N, size_t K, __half alpha, __half beta);

struct HgemmBetterReuseTuningCandidate {
    std::string name;
    HgemmFunction implementation;
    int bm;
    int bn;
    int bk;
};

#if LIB_HAS(HGEMM_CUBLAS)
float hgemm_cublas(const __half* A, const __half* B, __half* C, size_t M, size_t N, size_t K,__half alpha, __half beta);
#endif


#if LIB_HAS(HGEMM_FINAL)
float hgemm_final(const __half* A, const __half* B, __half* C, size_t M, size_t N, size_t K,__half alpha, __half beta);
#endif


#if LIB_HAS(HGEMM_BRUCE)
float hgemm_bruce(const __half* A, const __half* B, __half* C, size_t M, size_t N, size_t K,__half alpha, __half beta);
#endif

#if LIB_HAS(HGEMM_BRUCE2)
float hgemm_bruce2(const __half* A, const __half* B, __half* C, size_t M, size_t N, size_t K,__half alpha, __half beta);
#endif

#if LIB_HAS(HGEMM_BETTER_REUSE)
float hgemm_better_reuse(const __half* A, const __half* B, __half* C, size_t M, size_t N, size_t K,__half alpha, __half beta);
std::vector<HgemmBetterReuseTuningCandidate> hgemm_better_reuse_tuning_candidates();
#endif

#if LIB_HAS(HGEMM_GROK)
float hgemm_grok(const __half* A, const __half* B, __half* C, size_t M, size_t N, size_t K,__half alpha, __half beta);
#endif
