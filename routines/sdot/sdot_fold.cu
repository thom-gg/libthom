#include "sdot.h"

/* NOT THE OPTIMIZED ONE
This implementation is just kept to see the evolution of different implementations

For a clean, commented, and optimized one see:

------- sdot_warp.cu --------
*/


__global__ void sdotFoldKernel(int n, const float * x, const float * y, float * result) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    extern __shared__ float shmem[];

    if (idx < n) {
        shmem[threadIdx.x] = x[idx] * y[idx];
    }
    else {
        shmem[threadIdx.x] = 0.0f;
    }

    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shmem[threadIdx.x] += shmem[threadIdx.x + stride];
        }
        __syncthreads();
    }

    
    if (threadIdx.x == 0) {
        atomicAdd(result, shmem[0]);
    }
}

// considering incx and incy will always be 1 for simplicity
float sdot_fold(int n, const float * x, const float * y, float * result) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threadsPerBlock = 1024;
    int nbBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    size_t sizeOfSharedMemory = threadsPerBlock * sizeof(float);
    cudaEventRecord(start);
    sdotFoldKernel<<<nbBlocks, threadsPerBlock, sizeOfSharedMemory>>>(n,x,y,result);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float ms; 
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}