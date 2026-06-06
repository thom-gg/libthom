#include "sdot.h"


__global__ void sdotKernel(int n, const float * x, const float * y, float * result) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ float blockResult;
    if (threadIdx.x == 0) {blockResult = 0;}
    __syncthreads();

    if (idx < n) {
        float val = x[idx] * y[idx];
        atomicAdd(&blockResult, val);
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        atomicAdd(result, blockResult);
    }
}

// considering incx and incy will always be 1 for simplicity
float sdot_naive_atomics(int n, const float * x, const float * y, float * result) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int threadsPerBlock = 128;
    int nbBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    cudaEventRecord(start);
    sdotKernel<<<nbBlocks, threadsPerBlock>>>(n,x,y,result);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float ms; 
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}