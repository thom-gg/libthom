#include "routines.h"
#include "reduction.h"
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <cuda_runtime.h>
#include <time.h>
#include <unordered_map>


#define LOWER_BOUND -15.0
#define UPPER_BOUND 15.0

static float * cpuX;
static float * gpuX;
static float * gpuOutput;

using ReductionImpl = float (*)(const float*, float*, int);

static const std::unordered_map<std::string, ReductionImpl> reductionImpls = {
#if LIB_HAS(REDUCTION_WARP)
    {"warp", reduction_warp},
#endif
};

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
    if (argc < 4) {
        fprintf(stderr, "Missing reduction implementation. Usage: ./libthom <test|bench> reduction <impl>\n");
        exit(1);
    }

    char * impl = argv[3];
    auto implIt = reductionImpls.find(impl);
    if (implIt == reductionImpls.end()) {
        fprintf(stderr, "Unknown or unavailable reduction implementation: %s\n", impl);
        exit(1);
    }

    int N = 10000;
    allocate(N);
    resetVectors(N);

    float time = implIt->second(gpuX, gpuOutput, N);
    printf("time = %f\n", time);

    printf("Hello from reduction handler, run mode = %d\n", runMode);


}
