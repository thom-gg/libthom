#include "sdot.h"

/* NOT THE OPTIMIZED ONE
This implementation is just kept to see the evolution of different implementations

For a clean, commented, and optimized one see:

------- sdot_warp.cu --------
*/

__global__ void sdotFoldVecKernel(int n, const float * x, const float * y, float * result) {
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

    // once done with block-wise reduction, atomic add into global memory
    if (threadIdx.x == 0) {
        atomicAdd(result, shmem[0]);
    }
}

float sdot_fold_vectorized(int n, const float * x, const float * y, float * result) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threadsPerBlock = 1024;
    // dividing by 4 because each thread takes care of 4 elems
    int nbBlocks = ( (n/4) + threadsPerBlock - 1) / threadsPerBlock;

    size_t sizeOfSharedMemory = threadsPerBlock * sizeof(float);
    cudaEventRecord(start);
    sdotFoldVecKernel<<<nbBlocks, threadsPerBlock, sizeOfSharedMemory>>>(n,x,y,result);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float ms; 
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}