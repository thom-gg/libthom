#include <cuda_runtime.h>
#include "reduction.h"

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
__global__ void reduction_grid(const float * input, int N, int warpsPerBlock, float * resultsBlocks) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int warpId = tid / 32;
    int warpLane = tid % 32;
    int gridStride = gridDim.x * blockDim.x;
    
    extern __shared__ float shmem[];

    float localValue = 0.0f;

    int i;
    for (i = (bid * blockDim.x + tid) * 4; i<N-3; i+= gridStride * 4) {
        float4 vals = reinterpret_cast<const float4 *>(&input[i])[0];
        localValue += vals.x + vals.y + vals.z + vals.w;
    }

    for (int j = i; j<N; j+= gridStride) {
        localValue += input[j];
    }

    // warp lvl reduction
    localValue = warp_reduce(localValue);
    
    block_reduce(warpLane, warpId, warpsPerBlock, localValue, shmem, &(resultsBlocks[bid]));
}

__global__ void reduction_single_block(float * resultsBlocks, float * output, int nbBlocks) {
    // this kernel needs to be started on a single block
    float localValue = 0.0f;
    
    if (threadIdx.x < nbBlocks) {
        localValue = resultsBlocks[threadIdx.x];
    }
    int tid = threadIdx.x;
    extern __shared__ float shmem[];
    int warpsPerBlock = blockDim.x / 32;

    int warpId = threadIdx.x / 32;
    int warpLane = threadIdx.x % 32;

    localValue = warp_reduce(localValue);

    block_reduce(warpLane, warpId, warpsPerBlock, localValue, shmem, output);
}

float reduction_warp(const float* input,  float* output, int N) {
    cudaEvent_t start, stop;
    cudaError_t error;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int warpsPerBlock = 8;
    int threadsPerBlock = warpsPerBlock * 32;
    int nbBlocks = 256;
    size_t shmemSize = warpsPerBlock * sizeof(float);
    float * resultsBlocks;
    cudaMalloc((void**)&resultsBlocks, nbBlocks * sizeof(float));
    reduction_grid<<<nbBlocks, threadsPerBlock, shmemSize>>>(input, N, warpsPerBlock, resultsBlocks);

    error = cudaGetLastError();
    CUDA_CHECK(error);
    
    int threadsKernel2 = nbBlocks;
    size_t shmemKernel2 = (nbBlocks / 32) * sizeof(float);
    reduction_single_block<<<1,threadsKernel2,shmemKernel2>>>(resultsBlocks, output, nbBlocks);
    
    cudaFree(resultsBlocks);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;

}