#include "routines.h"
#include "reduction.h"
#include <cstddef>


#define LOWER_BOUND -15.0
#define UPPER_BOUND 15.0

static float * cpuX;
static float * gpuX;
static float * gpuOutput;
static cudaError_t err;

using ReductionImpl = float (*)(const float*, float*, int);


/* Registering implementations, only if they're compiled. They are compiled if we are using the correct Compute Capability*/
static const std::unordered_map<std::string, ReductionImpl> reductionImpls = {
#if LIB_HAS(REDUCTION_WARP)
    {"warp", reduction_warp},
#endif
};


void allocate(int N) {
    cpuX = (float*)malloc(N*sizeof(float));
    if (cpuX == NULL) {fprintf(stderr, "Error allocating cpuX\n");}

    err = cudaMalloc((void **)&gpuX, N*sizeof(float));
    CUDA_CHECK(err);
    err = cudaMalloc((void **)&gpuOutput, 1*sizeof(float));
    CUDA_CHECK(err);
}

void freeAll() {
    free(cpuX);
    cudaFree(gpuX);
    cudaFree(gpuOutput);
}

void resetVectors(int N) {
    for (int i = 0; i<N; i++) {cpuX[i] = getRdmDouble(LOWER_BOUND, UPPER_BOUND);}  

    err = cudaMemcpy(gpuX, cpuX, N*sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK(err);
    err = cudaMemset(gpuOutput, 0.0f, 1 * sizeof(float));
    CUDA_CHECK(err);
}

float getReferenceResults(int N) {
    float localValue = 0.0f;
    for (int i = 0; i<N; i++) {
        localValue += cpuX[i];
    }
    return localValue;
}

void compareToReference(int N) {
    // Copy GPU output to cpu
    float resultFromGpu;
    err = cudaMemcpy(&resultFromGpu, gpuOutput, 1*sizeof(float), cudaMemcpyDeviceToHost);
    CUDA_CHECK(err);
    float referenceResult = getReferenceResults(N);

    float relativeError = (abs(resultFromGpu - referenceResult) / abs(referenceResult));
    if (relativeError > 1e-4) {
        printf("ERROR: relativeError = %f, gpu=%f, cpu=%f\n", relativeError, resultFromGpu, referenceResult);
    }                
}

float getGflops(int N, float time_ms) {
    float flops = N - 1;
    float gflops = flops / (time_ms * 1e6f);
    return gflops;
}


void reduction_handler_test(int argc, char * argv[]) {
    /*
        ./libthom test reduction <impl> <size> <nb_iter> <check_result=1>
    */
    if (argc != 6 && argc != 7) {
        fprintf(stderr, "Missing arguments.\nUsage: ./libthom test reduction <impl> <size> <nb_iter> <check_result=1>\n");
        exit(1);
    }
    char * implName = argv[3];
    try {
        ReductionImpl impl = reductionImpls.at(implName);
        int size = atoi(argv[4]);
        int nb_iter = atoi(argv[5]);
        bool checkResult = true;
        if (argc == 7) {checkResult = atoi(argv[6]) == 1;}

        
        float sum_gflops = 0.0f;

        allocate(size);
        resetVectors(size);
        for (int i = 0; i<nb_iter; i++) {
            float time_ms = impl(gpuX, gpuOutput, size);
            float gflops = getGflops(size, time_ms);

            if (checkResult) {
                compareToReference(size);
            }

            printf("[%d]: %f GFLOP/s \n", i, gflops);
            sum_gflops += gflops;

            resetVectors(size);
            cudaDeviceSynchronize();
        }

        float avg_gflops = sum_gflops / nb_iter;
        printf("# GFLOP/s: %f\n", avg_gflops);
        freeAll();

    }
    catch (const std::out_of_range&) {
        fprintf(stderr, "Invalid implementation. Either it doesnt exist, or it's not available for the ARCH you've compiled\n");
        exit(1);
    }
}

/* BENCHMARKING */
static ReductionImpl selectedReductionImpl = nullptr;


void reduction_handler_bench(int argc, char* argv[]) {
    /* Handler called first that only sets the parameters up for benchmarking.
    The actual benchmark is invoked by starting nvbench in main.cpp*/
    char * implName = argv[3];
    try {
        selectedReductionImpl = reductionImpls.at(implName);
    }
    catch (const std::out_of_range&) {
        fprintf(stderr, "Invalid implementation. Either it doesnt exist, or it's not available for the ARCH you've compiled\n");
        exit(1);
    }
}

void reduction_handler(RunMode runMode, int argc, char * argv[]) {
    switch (runMode) {
        case RunMode::TEST:
            reduction_handler_test(argc, argv);
            break;
        case RunMode::BENCH:
            reduction_handler_bench(argc, argv);
            break;
    }

}
