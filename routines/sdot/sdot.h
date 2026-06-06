#pragma once

#include "../routines.h"

#if LIB_HAS(SDOT_NAIVE_ATOMICS)
float sdot_naive_atomics(int n, const float * x, const float * y, float * result);
#endif

#if LIB_HAS(SDOT_CUBLAS)
float sdot_cublas(int n, const float * x, const float * y, float * result);
#endif

#if LIB_HAS(SDOT_FOLD)
float sdot_fold(int n, const float * x, const float * y, float * result);
#endif

#if LIB_HAS(SDOT_FOLD_VECTORIZED)
float sdot_fold_vectorized(int n, const float * x, const float * y, float * result);
#endif

#if LIB_HAS(SDOT_2KERNEL)
float sdot_2kernel(int n, const float * x, const float * y, float * result);
#endif

#if LIB_HAS(SDOT_WARP)
float sdot_warp(int n, const float * x, const float * y, float * result);
#endif
