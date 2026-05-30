#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <cuda_runtime.h>
#include "reduction.h"

float reduction_thrust(const float* input,  float* output, int N) {
    cudaEvent_t start, stop;
    cudaError_t error;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    thrust::device_ptr<const float> begin(input);
    thrust::device_ptr<const float> end = begin + N;

     float resultHost = thrust::reduce(
        begin,
        end,
        0.0f
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Copy host result to gpu (just to comply with the custom kernels)
    cudaMemcpy(output, &resultHost, sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    float ms; 
    cudaEventElapsedTime(&ms, start, stop);
    return ms;

}