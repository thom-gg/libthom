#pragma once

#include "../routines.h"

#if LIB_HAS(SGEMV_NAIVE)
float sgemv_naive(int m, int n, const float * alpha, const float * A, int lda, const float * x, int incx, const float * beta, float * y, int incy, bool rowMajor);
#endif

