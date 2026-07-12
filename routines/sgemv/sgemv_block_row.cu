#include "sgemv.h"


/* NOT THE OPTIMIZED ONE
This implementation is just kept to see the evolution of different implementations

For clean, commented, and optimized ones see:

------- sgemv_vecto.cu (row major) --------
------- sgemv_col3_vecto.cu (col major) --------

*/

// 1 block per row of A (i.e per element of y)
// meaning we do intra-block reduction to get the final value of the cell of y
// - warp lvl reduction, each warp writes its value into a shmem array (of very small size, probably 4 warps per block smth like that)
// - shmem reduction
// tid 0 writes into GMEM y
__global__ void gemv_kernel_block_row(  int M, int N, const float * alpha, const float * A, int lda, 
                                        const float * x, int incx, const float * beta, float * y, int incy,
                                        int warpsPerBlock) {
    int tid = threadIdx.x;

    extern __shared__ float shmem[]; // size warpsPerBlock

    int rowId = blockIdx.x;

    float laneLocalDot = 0.0f;

    for (int i = tid; i<N; i+=blockDim.x) {
        laneLocalDot += A[rowId * lda + i] *  x[i];
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


float sgemv_block_row(int m, int n, const float * alpha, const float * A, int lda, const float * x, int incx, const float * beta, float * y, int incy, bool rowMajor) {
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
    gemv_kernel_block_row<<<nbBlocks, threadsPerBlock, shmemSize>>>(m, n, alpha, A, lda, x, incx, beta, y, incy, warpsPerBlock);    
    
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