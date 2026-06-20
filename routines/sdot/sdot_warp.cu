#include "sdot.h"

/* Performs warp-level reduction in registers, therefore without needing any synchronization */
// NOTE: using /= 2 generates a shift-right ptx instruction, we're not paying the cost of division here, 
// and it's more readable than "offset>>=1"
__device__ __forceinline__ float warp_reduce(float laneValue) {
    for (int offset = 16; offset > 0; offset /= 2) {
        laneValue += __shfl_down_sync(0xffffffff, laneValue, offset);
    }
    return laneValue;
};

/* When calling this function, threads in lane 0 (threadIdx.x % 32 == 0) are expected to hold the `warpValue` for their whole
warp, and the function will perform a block-wide reduction in SMEM and write the final value in `resultAddress`.  This is the same
behavior needed in both kernels for this implementation */
__device__ __forceinline__ void block_reduce(int laneIdx, int warpIdx, int nbWarps, float warpValue, float * smem, float * resultAddress) {
    // First thread of each warp writes it's value (representing its whole warp) in smem
    if (laneIdx == 0) { 
        smem[warpIdx] = warpValue;
    }
    __syncthreads();
    for (int stride = nbWarps / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] += smem[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        *resultAddress = smem[0];
    }
};

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
    localResult = warp_reduce(localResult);

    // Block-wide inter-warp reduction in SMEM
    block_reduce(lane, threadIdx.x / 32, nbWarpsInBlock, localResult, shmem, &(resultsBlocks[blockIdx.x]));
}


/* This kernel is launched on a single block and perform a reduction on a global memory array */
__global__ void sdot_reduction_1block(int nbBlocks, float * result, float * resultsBlocks) {
    extern __shared__ float shmem[];
    int idx = threadIdx.x; // there will be only one block
    int warpLane = idx % 32;
    int warpIdx = idx / 32;
    int nbWarps = blockDim.x / 32;
    
    // Load values from GMEM to registers
    float laneVal = 0.0f;
    for (int i = idx; i<nbBlocks; i += blockDim.x) {
        laneVal += resultsBlocks[i];
    }
    
    // Warp-level reduction
    // Each lane 0 has results for the whole warp
    laneVal = warp_reduce(laneVal);
    
    // Block-wide inter-warp reduction in SMEM
    block_reduce(warpLane, warpIdx, nbWarps, laneVal, shmem, result);
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