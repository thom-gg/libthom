#include "sdot.h"



float sdot_cublas(int n, const float * x, const float * y, float * result) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cublasSdot(cublasHandle, n, x, 1, y, 1, result);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float ms; 
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}