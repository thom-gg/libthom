#include "sdot.h"

/* NOT THE OPTIMIZED ONE
This implementation is just kept to see the evolution of different implementations

For a clean, commented, and optimized one see:

------- sdot_warp.cu --------
*/


__global__ void sdotFoldVecKernel1(int n, const float * x, const float * y, float * resultsBlocks) {
    // each thread is going to take care of 4 elems
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 4;

    extern __shared__ float shmem[];

    // when possible use vectorized reads
    if (idx < n && (idx+3)<n) {
        float4 x_vec = (reinterpret_cast<const float4 *>(&(x[idx])))[0];
        float4 y_vec = (reinterpret_cast<const float4 *>(&(y[idx])))[0];

        float val = (x_vec.x * y_vec.x) + (x_vec.y * y_vec.y) + (x_vec.z * y_vec.z) + (x_vec.w * y_vec.w); 

        shmem[threadIdx.x] = val;
    }
    // take care of edge case and read the (maximum 3) last values manually
    // (last thread, only one thread in the whole grid)
    else if (idx < n) {
        float val = 0.0;
        for (int i = idx; i<n; i++) {
            val += x[i] * y[i];
        }
        shmem[threadIdx.x] = val;
        
    }
    // for the last block, which might have useless threads
    else {
        shmem[threadIdx.x] = 0.0f;
    }
    
    __syncthreads();

    // After loading values per thread, perform block-wise reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shmem[threadIdx.x] += shmem[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        resultsBlocks[blockIdx.x] = shmem[0];
    }
}

__global__ void sdotFoldVecKernel2(int nbBlocks, float * result, float * resultsBlocks) {
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
float sdot_2kernel(int n, const float * x, const float * y, float * result) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    
    int threadsPerBlock = 1024;
    // dividing by 4 because each thread takes care of 4 elems
    int nbElemsPerBlock = threadsPerBlock * 4;
    int nbBlocks = (n + nbElemsPerBlock - 1) / nbElemsPerBlock;
    float * resultsBlocks;
    cudaMalloc((void**)&resultsBlocks, nbBlocks * sizeof(float));
    size_t sizeOfSharedMemory = threadsPerBlock * sizeof(float);
    sdotFoldVecKernel1<<<nbBlocks, threadsPerBlock, sizeOfSharedMemory>>>(n,x,y,resultsBlocks);
    
    // lancer le deuxième kernel
    int threadsKernel2 = fmin(nbBlocks, 1024);
    sizeOfSharedMemory = threadsKernel2 * sizeof(float);
    sdotFoldVecKernel2<<<1, threadsKernel2, sizeOfSharedMemory>>>(nbBlocks, result, resultsBlocks);
    
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaFree(resultsBlocks);
    float ms; 
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}