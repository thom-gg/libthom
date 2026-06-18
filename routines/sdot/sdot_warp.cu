#include "sdot.h"


/* This kernel performs a reduction using warp-level methods and then shared memory for inter-warp reduction, and each block
writes its result in GMEM (for a second kernel to later perform reduction of block results)*/
__global__ void sdot_dot_product_block(int n, const float * x, const float * y, float * resultsBlocks) {
    // each thread is going to take care of 4 elems
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
    int lane = threadIdx.x % 32;
    int nbWarpsInBlock = blockDim.x / 32;
    

    extern __shared__ float shmem[];
    float localResult = 0.0f; // going to hold a single thread's result

      // when possible use vectorized reads
    if (idx < n && (idx+3)<n) {
        float4 x_vec = (reinterpret_cast<const float4 *>(&(x[idx])))[0];
        float4 y_vec = (reinterpret_cast<const float4 *>(&(y[idx])))[0];

        float val = (x_vec.x * y_vec.x) + (x_vec.y * y_vec.y) + (x_vec.z * y_vec.z) + (x_vec.w * y_vec.w); 

        localResult= val;
    }
    // take care of edge case and read the (maximum 3) last values manually
    // (the last thread of the block takes this branch, only one thread in the whole grid)
    else if (idx < n) {
        float val = 0.0;
        for (int i = idx; i<n; i++) {
            val += x[i] * y[i];
        }
       localResult = val;
        
    }
    // for the last block, which might have useless threads
    else {
        localResult = 0.0f;
    }
    

    // Perform warp level reduction to get one value per warp
    for (int offset = 16; offset > 0; offset /= 2) {
        localResult += __shfl_down_sync(0xffffffff, localResult, offset);
    }

    if (lane == 0) { // only first thread of each warp writes in shmem
        shmem[threadIdx.x / 32] = localResult;
    }

    __syncthreads();

    
    // After loading values per thread, perform block-wise reduction
    for (int stride = nbWarpsInBlock / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shmem[threadIdx.x] += shmem[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        resultsBlocks[blockIdx.x] = shmem[0];
    }
}

/* This kernel is launched on a single block and perform a reduction on a global memory array */
__global__ void sdot_reduction_1block(int nbBlocks, float * result, float * resultsBlocks) {
    // NOTE: TODO:
    // as i'm cleaning up this code i realize this reduction could be warp-level first then use shmem if there are more than
    // 32 elements to reduce. I don't have a GPU rn to test it out so i'm not implementing it yet but this should be done.
    int idx = threadIdx.x; // there will be only one block

    extern __shared__ float shmem[];
    shmem[idx] = 0;
    // Every thread loads it's corresponding element into shmem. If there are more elements than threads, perform the addition
    // as we are iterating through GMEM
    for (int i = idx; i<nbBlocks; i += blockDim.x) {
        shmem[idx] += resultsBlocks[i];
    }


    __syncthreads();

    int logicalSize = blockDim.x;
    while (logicalSize > 1) {
        int stride = (logicalSize+1) >> 1;
        if (idx < stride && (idx+stride) < logicalSize) {
            shmem[idx] += shmem[idx + stride];
        }
        __syncthreads();
        logicalSize = stride;
    }   

    if (idx == 0) {
        *result = shmem[0];
    }
}
float sdot_warp(int n, const float * x, const float * y, float * result) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    
    int nbBlocks;
    int threadsPerBlock;
    

    threadsPerBlock = 1024;
    int nbElemsPerBlock = threadsPerBlock * 4;
    nbBlocks = (n + nbElemsPerBlock - 1) / nbElemsPerBlock;

    // Allocating a GMEM array for each block to write its result
    float * resultsBlocks;
    cudaMalloc((void**)&resultsBlocks, nbBlocks * sizeof(float));

    // Warps will reduce themselves using warp-reduction to a single value, so we only need one value per warp in shared memory to
    // make a final block reduction before writing to GMEM
    size_t sizeOfSharedMemory = (threadsPerBlock / 32) * sizeof(float); 

    // Launching first kernel
    sdot_dot_product_block<<<nbBlocks, threadsPerBlock, sizeOfSharedMemory>>>(n,x,y,resultsBlocks);

    // Second kernel need as many threads as elements to reduce, or 1024 at maximum
    int threadsKernel2 = fmin(nbBlocks, 1024);
    // need as many shmem slots as threads in the kernel
    sizeOfSharedMemory = threadsKernel2 * sizeof(float);
    sdot_reduction_1block<<<1, threadsKernel2, sizeOfSharedMemory>>>(nbBlocks, result, resultsBlocks);
    
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaFree(resultsBlocks);
    float ms; 
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}