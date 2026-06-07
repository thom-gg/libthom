#include "dgemm.h"


float dgemm_cublas_tc(const double* A, const double* B, double* C, size_t M, size_t N, size_t K,double alpha, double beta) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cublasGemmEx(dgemm::cublasHandle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N,M,K,
            &alpha,
            B,CUDA_R_64F ,N,
            A,CUDA_R_64F,K,
            &beta,
            C,CUDA_R_64F,N,
            CUBLAS_COMPUTE_64F,
            CUBLAS_GEMM_DEFAULT
        );

    cudaEventRecord(stop);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
    }

    cudaEventSynchronize(stop);
    float ms; 
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}