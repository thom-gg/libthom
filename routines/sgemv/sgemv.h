#pragma once

#include "../routines.h"

#if LIB_HAS(SGEMV_NAIVE)
float sgemv_naive(int m, int n, const float * alpha, const float * A, int lda, const float * x, int incx, const float * beta, float * y, int incy, bool rowMajor);
#endif

#if LIB_HAS(SGEMV_BLOCK_ROW)
float sgemv_block_row(int m, int n, const float * alpha, const float * A, int lda, const float * x, int incx, const float * beta, float * y, int incy, bool rowMajor);
#endif

#if LIB_HAS(SGEMV_COL1_NAIVE)
float sgemv_col1_naive(int m, int n, const float * alpha, const float * A, int lda, const float * x, int incx, const float * beta, float * y, int incy, bool rowMajor);
#endif

#if LIB_HAS(SGEMV_COL2_SHMEM)
float sgemv_col2_shmem(int m, int n, const float * alpha, const float * A, int lda, const float * x, int incx, const float * beta, float * y, int incy, bool rowMajor);
#endif

#if LIB_HAS(SGEMV_COL3_VECTO)
float sgemv_col3_vecto(int m, int n, const float * alpha, const float * A, int lda, const float * x, int incx, const float * beta, float * y, int incy, bool rowMajor);
#endif

#if LIB_HAS(SGEMV_COL4_TRANS)
float sgemv_col4_trans(int m, int n, const float * alpha, const float * A, int lda, const float * x, int incx, const float * beta, float * y, int incy, bool rowMajor);
#endif

#if LIB_HAS(SGEMV_CUBLAS)
float sgemv_cublas(int m, int n, const float * alpha, const float * A, int lda, const float * x, int incx, const float * beta, float * y, int incy, bool rowMajor);
#endif

#if LIB_HAS(SGEMV_VECTO)
float sgemv_vecto(int m, int n, const float * alpha, const float * A, int lda, const float * x, int incx, const float * beta, float * y, int incy, bool rowMajor);
#endif

