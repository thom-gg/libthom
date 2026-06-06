#include "sgemv.h"


// 1 block per row of A (i.e per element of y)
// meaning we do intra-block reduction to get the final value of the cell of y
// - warp lvl reduction, each warp writes its value into a shmem array (of very small size, probably 4 warps per block smth like that)
// - shmem reduction
// tid 0 writes into GMEM y
// but we use vectorisation to read from A or x
__global__ void gemv_kernel_vecto(int M, int N, const float * alpha, const float * A, int lda, const float * x, int incx, 
                                const float * beta, float * y, int incy, int warpsPerBlock) {
    int tid = threadIdx.x;

    extern __shared__ float shmem[]; // size warpsPerBlock

    int rowId = blockIdx.x;

    float laneLocalDot = 0.0f;

    int i;
    for (i = tid*4; i<=N-4; i+=blockDim.x*4) {
        float4 aVals = reinterpret_cast<const float4 *>(&A[rowId * lda + i])[0];
        float4 xVals = reinterpret_cast<const float4 *>(&x[i])[0];
        laneLocalDot += aVals.x * xVals.x;
        laneLocalDot += aVals.y * xVals.y;
        laneLocalDot += aVals.z * xVals.z;
        laneLocalDot += aVals.w * xVals.w;
    }
    // take care of last 3 elems
    for (int j = i; j < N; j += blockDim.x) {
        laneLocalDot += A[rowId * lda + j] * x[j];
    }

    // warp lvl reduction
    for (int offset = 16; offset >0; offset >>=1) {
        laneLocalDot += __shfl_down_sync(0xffffffff, laneLocalDot, offset);
    }

    if (tid % 32 == 0) { // lane 0 writes into shmem
        shmem[tid / 32] = laneLocalDot;
    }

    __syncthreads();
    // block lvl reduction
    for (int stride = warpsPerBlock / 2; stride > 0; stride>>=1) {
        if (tid < stride) {
            shmem[tid] += shmem[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        y[rowId] = *(alpha) * shmem[tid] + *(beta) * y[rowId];

    }


    
    
}


float sgemv_vecto(int m, int n, const float * alpha, const float * A, int lda, const float * x, int incx, const float * beta, float * y, int incy, bool rowMajor) {
    if (!rowMajor) {
        fprintf(stderr, "Called a row major kernel on a COLUMN major execution; Stopping\n");
        exit(1);
    }
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    
  // chosen warpsPerBlock has to be a power of 2 for the block-lvl reduction to work;
    int warpsPerBlock = 4;
    int threadsPerBlock =  warpsPerBlock * 32;
    int nbBlocks = m; // 1 block per row
    size_t shmemSize = warpsPerBlock * sizeof(float); // for the block reduction

    gemv_kernel_vecto<<<nbBlocks, threadsPerBlock, shmemSize>>>(m, n, alpha, A, lda, x, incx, beta, y, incy, warpsPerBlock);    
    
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