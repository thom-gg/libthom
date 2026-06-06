#include "sgemv.h"



float sgemv_cublas(int m, int n, const float * alpha, const float * A, int lda, const float * x, int incx, const float * beta, float * y, int incy, bool rowMajor) {
    cublasSetPointerMode(sgemv::cublasHandle, CUBLAS_POINTER_MODE_DEVICE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cublasSgemv(sgemv::cublasHandle, rowMajor ? CUBLAS_OP_T : CUBLAS_OP_N, n,m, alpha, A, lda, x, incx, beta, y, incy);
    CUDA_CHECK(cudaGetLastError());
    
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float ms; 
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}