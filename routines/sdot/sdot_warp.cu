#include "sdot.h"



__global__ void sdotDotProductWarpReductionWITHOUTSTRIDE(int n, const float * x, const float * y, float * resultsBlocks) {
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
    // (last thread, only one thread in the whole grid)
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


__global__ void sdotDotProductWarpReduction(int n, const float * x, const float * y, float * resultsBlocks) {
    // each thread is going to take care of 4 elems
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
    int lane = threadIdx.x % 32;
    int nbWarpsInBlock = blockDim.x / 32;
    

    extern __shared__ float shmem[];
    float localResult = 0.0f; // going to hold a single thread's result

    for (int currentIndex = idx; currentIndex < n; currentIndex += blockDim.x * gridDim.x * 4) {
        // when possible use vectorized reads
        if ((currentIndex+3)<n) {
            float4 x_vec = (reinterpret_cast<const float4 *>(&(x[currentIndex])))[0];
            float4 y_vec = (reinterpret_cast<const float4 *>(&(y[currentIndex])))[0];
    
            float val = (x_vec.x * y_vec.x) + (x_vec.y * y_vec.y) + (x_vec.z * y_vec.z) + (x_vec.w * y_vec.w); 
    
            localResult += val;
        }
        // take care of edge case and read the (maximum 3) last values manually
        // (last thread, only one thread in the whole grid)
        else {
            float val = 0.0;
            for (int i = currentIndex; i<n; i++) {
                val += x[i] * y[i];
            }
            localResult += val;
        }
    }
    // for the last block, which might have useless threads
    if (idx >= n) {
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

__global__ void sdot1BlockReduction(int nbBlocks, float * result, float * resultsBlocks) {
    int idx = threadIdx.x; // there will be only one block

    extern __shared__ float shmem[];
    shmem[idx] = 0;
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
    
    // // new way of launching grid, for version with grid stride
    // int nbSM = 30; // for my 2060
    // threadsPerBlock = 256;
    // nbBlocks = nbSM *3;

    // legacy way of launching grid
    threadsPerBlock = 1024;
    int nbElemsPerBlock = threadsPerBlock * 4;
    nbBlocks = (n + nbElemsPerBlock - 1) / nbElemsPerBlock;

    float * resultsBlocks;
    cudaMalloc((void**)&resultsBlocks, nbBlocks * sizeof(float));
    size_t sizeOfSharedMemory = (threadsPerBlock / 32) * sizeof(float); // we need only one float per warp in shared memory
    // sdotDotProductWarpReduction<<<nbBlocks, threadsPerBlock, sizeOfSharedMemory>>>(n,x,y,resultsBlocks);
    sdotDotProductWarpReductionWITHOUTSTRIDE<<<nbBlocks, threadsPerBlock, sizeOfSharedMemory>>>(n,x,y,resultsBlocks);
    // lancer le deuxième kernel
    int threadsKernel2 = fmin(nbBlocks, 1024);
    sizeOfSharedMemory = threadsKernel2 * sizeof(float);
    sdot1BlockReduction<<<1, threadsKernel2, sizeOfSharedMemory>>>(nbBlocks, result, resultsBlocks);
    
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaFree(resultsBlocks);
    float ms; 
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}