#include "hgemm.h"


float hgemm_cublas(const __half* A, const __half* B, __half* C, size_t M, size_t N, size_t K,__half alpha, __half beta) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);



    cublasGemmEx(hgemm::cublasHandle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N,M,K,
            &alpha,
            B,CUDA_R_16F ,N,
            A,CUDA_R_16F,K,
            &beta,
            C,CUDA_R_16F,N,
            CUBLAS_COMPUTE_16F_PEDANTIC,
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