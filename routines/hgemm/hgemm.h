#pragma once

#include "../routines.h"
#include "cuda_fp16.h"

#if LIB_HAS(HGEMM_CUBLAS)
float hgemm_cublas(const __half* A, const __half* B, __half* C, size_t M, size_t N, size_t K,__half alpha, __half beta);
#endif

