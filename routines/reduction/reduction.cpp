#include "routines.h"
#include "reduction.h"



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
    if (err != cudaSuccess) {fprintf(stderr, "Error allocating gpuX %s\n", cudaGetErrorString(err));}
    err = cudaMalloc((void **)&gpuOutput, 1*sizeof(float));
    if (err != cudaSuccess) {fprintf(stderr, "Error allocating gpuOutput %s\n", cudaGetErrorString(err));}
}

void freeAll() {
    free(cpuX);
    cudaFree(gpuX);
    cudaFree(gpuOutput);
}

void resetVectors(int N) {
    for (int i = 0; i<N; i++) {cpuX[i] = getRdmFloat();}  

    cudaError_t err = cudaMemcpy(gpuX, cpuX, N*sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {fprintf(stderr, "Error copying to gpuX %s\n", cudaGetErrorString(err));}
    err = cudaMemset(gpuOutput, 0.0f, 1 * sizeof(float));
    if (err != cudaSuccess) {fprintf(stderr, "Error memsetting to gpuOutput: %s\n", cudaGetErrorString(err));}
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
    cudaMemcpy(&resultFromGpu, gpuOutput, 1*sizeof(float), cudaMemcpyDeviceToHost);
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

void reduction_handler_bench(int argc, char* argv[]) {

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
