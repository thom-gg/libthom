#include <cuda_runtime.h>
#include "reduction.h"

// Reduction with the add operator on the input
__global__ void first_kernel(const float * input, int N, int warpsPerBlock, float * resultsBlocks) {
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
    for (int offset = 16; offset > 0; offset /= 2) {
        localValue += __shfl_down_sync(0xFFFFFFFF, localValue, offset);
    }

    if (warpLane == 0) {
        shmem[warpId] = localValue;
    }
    __syncthreads();

    
    // block lvl reduction, made by a single warp to avoid synchronizations
    if (warpId == 0) {
        float warpVal = 0.0f;
        if (warpLane < warpsPerBlock) {
            warpVal = shmem[warpLane];
        }
        for (int offset = 16; offset > 0; offset /= 2) {
            warpVal += __shfl_down_sync(0xFFFFFFFF, warpVal, offset);
        }

        if (warpLane == 0) {
            resultsBlocks[bid] = warpVal;
        }
    }
    

  
}

__global__ void gmem_red_single_block(float * resultsBlocks, float * output, int nbBlocks) {
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
    for (int offset = 16; offset > 0; offset /=2) {
        localValue += __shfl_down_sync(0xFFFFFFFF, localValue, offset);
    }

    if (warpLane == 0) {
        shmem[warpId] = localValue;
    }
    __syncthreads();
    // block lvl reduction
    for (int offset = warpsPerBlock / 2; offset > 0; offset /= 2) {
        if (tid < offset) {
            shmem[tid] += shmem[tid + offset];
        } 
        __syncthreads();
    }

    if (tid == 0) {
        *output = shmem[0];
    }
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
    first_kernel<<<nbBlocks, threadsPerBlock, shmemSize>>>(input, N, warpsPerBlock, resultsBlocks);
    error = cudaGetLastError();
    CUDA_CHECK(error);
    int threadsKernel2 = nbBlocks;
    size_t shmemKernel2 = (nbBlocks / 32) * sizeof(float);
    gmem_red_single_block<<<1,threadsKernel2,shmemKernel2>>>(resultsBlocks, output, nbBlocks);
    
    cudaFree(resultsBlocks);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms; 
    cudaEventElapsedTime(&ms, start, stop);
    return ms;

}