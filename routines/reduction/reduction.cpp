#include "routines.h"
#include "reduction.h"
#include <cstddef>
#include <fstream>
#include <nvtx3/nvToolsExt.h>

#define BENCH_RUNS 10
#define WARMUP_RUNS 5
const std::vector<std::vector<int>> SIZES = {{10000}, {100'000}, {1'000'000}, {10'000'000}, {100'000'000}, {200'000'000}, {300'000'000}, {500'000'000}};


#define LOWER_BOUND -3.0
#define UPPER_BOUND 3.0

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
#if LIB_HAS(REDUCTION_THRUST)
    {"thrust", reduction_thrust},
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
    CUDA_CHECK(err);   
    cudaDeviceSynchronize();
    err = cudaMemcpy(cpuX, gpuX, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}

float getReferenceResults(int N) {
    double localValue = 0.0;
    for (int i = 0; i<N; i++) {
        localValue += (double)cpuX[i];
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
float runner(ReductionImpl impl, std::vector<int> sizes, int nb_runs, int nb_warmup_runs, bool checkResult, bool printEach, int argc, char * argv[]) {
    float sum_gflops = 0.0f;
    allocate(sizes[0]);
    resetVectors(sizes[0]);

    for (int i = -1*nb_warmup_runs; i<nb_runs; i++) {
        float time_ms = impl(gpuX, gpuOutput, sizes[0]);

        
        if (checkResult) {
            compareToReference(sizes[0]);
        }
        
        if (i >= 0) {

            float gflops = getGflops(sizes[0], time_ms);

            if (printEach) {
                printf("[internal][iter %d]: %f GFLOP/s \n", i, gflops);
            }
            sum_gflops += gflops;
        }


        resetVectors(sizes[0]);
        cudaDeviceSynchronize();
    }

    float avg_gflops = sum_gflops / nb_runs;

    freeAll();

    return avg_gflops;
}


/* BENCHMARKING */

void exportCSV(const std::map<std::vector<int>, float>& results,
               const std::string& filename)
{
    std::ofstream file(filename);

    file << "size,gflops\n";

    for (const auto& [sizes, gflops] : results) {
        file << sizes[0] << "," << gflops << "\n";
    }
}

void reduction_handler(RunMode runMode, int argc, char * argv[]) {
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);


    switch (runMode) {
        case RunMode::TEST:
            Handlers<ReductionImpl, 1, BENCH_RUNS, WARMUP_RUNS>::test_handler(argc, argv, reductionImpls, runner);
            break;
        case RunMode::BENCH:
            Handlers<ReductionImpl, 1, BENCH_RUNS,WARMUP_RUNS>::bench_handler(argc, argv, reductionImpls, runner, SIZES, exportCSV);
            break;
    }

    curandDestroyGenerator(gen);

}
