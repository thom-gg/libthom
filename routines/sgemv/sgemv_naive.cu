#include "sgemv.h"


// 1 thread per row of A (i.e per element of y)
__global__ void gemv_kernel(int M, int N, const float * alpha, const float * A, int lda, const float * x, int incx, const float * beta, float * y, int incy) {
    int tid = threadIdx.x;
    int globalIdx = tid + blockIdx.x * blockDim.x;
    int rowId = globalIdx;

    if (rowId < M) {
        float dot = 0.0f;
        for (int i = 0; i<N; i++) {
            dot += A[rowId * lda + i] *  x[i];
        }
        y[rowId] = *(alpha) * dot + *(beta) * y[rowId];
    }
    
}


float sgemv_naive(int m, int n, const float * alpha, const float * A, int lda, const float * x, int incx, const float * beta, float * y, int incy, bool rowMajor) {
    if (!rowMajor) {
        fprintf(stderr, "Called a row major kernel on a COLUMN major execution; Stopping\n");
        exit(1);
    }
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);


    
    int threadsPerBlock = 512;
    int nbBlocks = (m + threadsPerBlock - 1) / threadsPerBlock;
    gemv_kernel<<<nbBlocks, threadsPerBlock>>>(m, n, alpha, A, lda, x, incx, beta, y, incy);    
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
    }
    
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float ms; 
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}