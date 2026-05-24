#include "routines.h"
#include "reduction.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>


#define LOWER_BOUND -15.0
#define UPPER_BOUND 15.0

static float * cpuX;
static float * gpuX;
static float * gpuOutput;

double getRdmFloat() {
    return LOWER_BOUND + ((float)rand() / RAND_MAX) * (UPPER_BOUND-LOWER_BOUND);
}

void allocate(int N) {
    cpuX = (float*)malloc(N*sizeof(float));
    if (cpuX == NULL) {fprintf(stderr, "Error allocating cpuX\n");}

    cudaError_t err;
    err = cudaMalloc((void **)&gpuX, N*sizeof(float));
    if (err != cudaSuccess) {fprintf(stderr, "Error allocating gpuX\n");}
    err = cudaMalloc((void **)&gpuOutput, 1*sizeof(float));
    if (err != cudaSuccess) {fprintf(stderr, "Error allocating gpuOutput\n");}
}

void resetVectors(int N) {
    for (int i = 0; i<N; i++) {cpuX[i] = getRdmFloat();}  

    cudaError_t err = cudaMemcpy(gpuX, cpuX, N*sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {fprintf(stderr, "Error copying to gpuX %s\n", cudaGetErrorString(err));}
}

void reduction_handler(RunMode runMode, int argc, char * argv[]) {
    int N = 10000;
    allocate(N);
    resetVectors(N);

    float time = reduction_warp(gpuX, gpuOutput, N);
    printf("time = %f\n", time);
    printf("Hello from reduction handler, run mode = %d\n", runMode);


}
