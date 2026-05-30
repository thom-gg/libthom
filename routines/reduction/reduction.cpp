#include "routines.h"
#include "reduction.h"
#include <cstddef>
#include <fstream>
#include <map>
#include <nvtx3/nvToolsExt.h>
#include <curand.h>
#include <curand_kernel.h>

#define BENCH_RUNS 10
#define WARMUP_RUNS 5
const int SIZES[] = {10000, 100'000, 1'000'000, 10'000'000, 100'000'000, 200'000'000};


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

curandGenerator_t gen;

void resetVectors(int N) {
    CURAND_CHECK(curandGenerateUniform(gen, gpuX, N));
    err = cudaMemset(gpuOutput, 0, sizeof(float));
    CUDA_CHECK(err);   cudaDeviceSynchronize();
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

/* Runner that can be called either by  the test handler or benchmark handler, and will return the average gflops */
float runner(ReductionImpl impl, int size, int nb_runs, int nb_warmup_runs, bool checkResult, bool printEach) {
    float sum_gflops = 0.0f;
    allocate(size);
    resetVectors(size);

    for (int i = -1*nb_warmup_runs; i<nb_runs; i++) {
        float time_ms = impl(gpuX, gpuOutput, size);

        
        if (checkResult) {
            compareToReference(size);
        }
        
        if (i >= 0) {

            float gflops = getGflops(size, time_ms);

            if (printEach) {
                printf("[internal][iter %d]: %f GFLOP/s \n", i, gflops);
            }
            sum_gflops += gflops;
        }


        resetVectors(size);
        cudaDeviceSynchronize();
    }

    float avg_gflops = sum_gflops / nb_runs;

    freeAll();

    return avg_gflops;
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

        
        float gflops = runner(impl, size, nb_iter, 0, checkResult, true);
        printf("# GFLOP/s: %f\n", gflops);

    }
    catch (const std::out_of_range&) {
        fprintf(stderr, "Invalid implementation. Either it doesnt exist, or it's not available for the ARCH you've compiled\n");
        exit(1);
    }
}

/* BENCHMARKING */

void exportCSV(const std::map<int, float>& results,
               const std::string& filename)
{
    std::ofstream file(filename);

    file << "size,gflops\n";

    for (const auto& [size, gflops] : results) {
        file << size << "," << gflops << "\n";
    }
}

void reduction_handler_bench(int argc, char* argv[]) {
    /*
        ./libthom bench reduction impl
    */
    if (argc != 4) {
        fprintf(stderr, "Missing arguments.\nUsage: ./libthom bench reduction <impl>\n");
        exit(1);
    }
    char * implName = argv[3];
    std::map<int, float> results;

    try {
        ReductionImpl impl = reductionImpls.at(implName);
        for (int size : SIZES) {
            printf("\nStarting benchmark for size %d...\n", size);
            float gflops = runner(impl, size, BENCH_RUNS, WARMUP_RUNS, false, true);
            results[size] = gflops;
            printf("[size: %d] ==> %f\n", size, gflops);
        }
        std::string filename = std::string("bench_") + implName + ".csv";
        exportCSV(results, filename);
    }
    catch (const std::out_of_range&) {
        fprintf(stderr, "Invalid implementation. Either it doesnt exist, or it's not available for the ARCH you've compiled\n");
        exit(1);
    }
}

void reduction_handler(RunMode runMode, int argc, char * argv[]) {
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);


    switch (runMode) {
        case RunMode::TEST:
            reduction_handler_test(argc, argv);
            break;
        case RunMode::BENCH:
            reduction_handler_bench(argc, argv);
            break;
    }

    curandDestroyGenerator(gen);

}
