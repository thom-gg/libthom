#include "w4a16.h"

/* Baseline: plain fp16 GEMM on the dequantized weights (W_dequant), so it reads 2*K*N bytes of weights
instead of K*N/2. Same contract as the W4A16 kernels: fp16 inputs and output, fp32 accumulation. */
float w4a16_cublas(const W4A16Problem& problem) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Row major C = A * W computed as column major C^T = W^T * A^T
    CUBLAS_CHECK(cublasGemmEx(w4a16::cublasHandle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            problem.N, problem.M, problem.K,
            &alpha,
            problem.W_dequant, CUDA_R_16F, problem.N,
            problem.A, CUDA_R_16F, problem.K,
            &beta,
            problem.C, CUDA_R_16F, problem.N,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT
        ));

    cudaEventRecord(stop);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
    }

    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}
