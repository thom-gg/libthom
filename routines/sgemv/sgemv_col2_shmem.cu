#include "sgemv.h"


/* NOT THE OPTIMIZED ONE
This implementation is just kept to see the evolution of different implementations

For clean, commented, and optimized ones see:

------- sgemv_vecto.cu (row major) --------
------- sgemv_col3_vecto.cu (col major) --------

*/

// 1 thread per row of A (coalesced accesseses to both A and x)
__global__ void col2_shmem_kernel(int M, int N, const float * alpha, const float * A, int lda, const float * x, int incx, const float * beta, float * y, int incy) {
    int tid = threadIdx.x;
    int globalIdx = tid + blockIdx.x * blockDim.x;
    int rowId = globalIdx;

    extern __shared__ float shmem[];

    for (int i = tid; i < N; i+= blockDim.x) {
        shmem[i] = x[i * incx];
    }
    __syncthreads();

    if (rowId < M) {
        float dot = 0.0f;
        for (int i = 0; i<N; i++) {
            dot += A[rowId + lda * i] *  shmem[i];
        }
        y[rowId] = *(alpha) * dot + *(beta) * y[rowId];
    }
    
}


float sgemv_col2_shmem(int m, int n, const float * alpha, const float * A, int lda, const float * x, int incx, const float * beta, float * y, int incy, bool rowMajor) {
    if (rowMajor) {
        fprintf(stderr, "Called a column major kernel on a ROW major execution; Stopping\n");
        exit(1);
    }
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int threadsPerBlock = 512;
    int nbBlocks = (m + threadsPerBlock - 1) / threadsPerBlock;
    size_t shmemSize = n * sizeof(float); // Shared memory size for the whole x vector
    col2_shmem_kernel<<<nbBlocks, threadsPerBlock, shmemSize>>>(m, n, alpha, A, lda, x, incx, beta, y, incy);
  
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