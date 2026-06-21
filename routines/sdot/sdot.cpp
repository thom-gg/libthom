#include "sdot.h"
#include <cstddef>
#include <fstream>
#include <nvtx3/nvToolsExt.h>
#include <curand.h>
#include <curand_kernel.h>

#define BENCH_RUNS 10
#define WARMUP_RUNS 5


#define LOWER_BOUND -5.0
#define UPPER_BOUND 5.0



namespace sdot {
    cublasHandle_t cublasHandle;
    static const std::vector<std::vector<int>> SIZES = {{10000}, {100'000}, {1'000'000}, {10'000'000}, {100'000'000}, {200'000'000}, {300'000'000}, {500'000'000}};
    
    static float * cpuX;
    static float * cpuY;
    
    static float * gpuX;
    static float * gpuY;
    static float * gpuOutput;


    curandGenerator_t gen;
    static cudaError_t err;
    
    using SdotImpl = float (*)(int, const float*, const float*, float*);
    
    
    /* Registering implementations, only if they're compiled. They are compiled if we are using the correct Compute Capability*/
    static const std::unordered_map<std::string, SdotImpl> sdotImpls = {
    #if LIB_HAS(SDOT_NAIVE_ATOMICS)
        {"naive_atomics", sdot_naive_atomics},
    #endif
    #if LIB_HAS(SDOT_CUBLAS)
        {"cublas", sdot_cublas},
    #endif
    #if LIB_HAS(SDOT_FOLD)
        {"fold", sdot_fold},
    #endif
    #if LIB_HAS(SDOT_FOLD_VECTORIZED)
        {"fold_vectorized", sdot_fold_vectorized},
    #endif
    #if LIB_HAS(SDOT_2KERNEL)
        {"2kernel", sdot_2kernel},
    #endif
    #if LIB_HAS(SDOT_WARP)
        {"warp", sdot_warp},
    #endif
    };
    
    void allocate(int N) {
        cpuX = (float*)malloc(N*sizeof(float));
        if (cpuX == NULL) {fprintf(stderr, "Error allocating cpuX\n");}
    
        cpuY = (float*)malloc(N*sizeof(float));
        if (cpuY == NULL) {fprintf(stderr, "Error allocating cpuX\n");}
    
        err = cudaMalloc((void **)&gpuX, N*sizeof(float));
        CUDA_CHECK(err);
        err = cudaMalloc((void **)&gpuY, N*sizeof(float));
        CUDA_CHECK(err);
        err = cudaMalloc((void **)&gpuOutput, 1*sizeof(float));
        CUDA_CHECK(err);
    }
    
    void freeAll() {
        free(cpuX);
        free(cpuY);
        cudaFree(gpuX);
        cudaFree(gpuY);
        cudaFree(gpuOutput);
    }
    
    
    void resetVectors(int N) {
        // fill randomly using cuRAND directly on gpu
        CURAND_CHECK(curandGenerateUniform(gen, gpuX, N));
        CURAND_CHECK(curandGenerateUniform(gen, gpuY, N));
        err = cudaMemset(gpuOutput, 0, sizeof(float));
        CUDA_CHECK(err);   
        cudaDeviceSynchronize();
        // copy on cpu for reference computations
        err = cudaMemcpy(cpuX, gpuX, N * sizeof(float), cudaMemcpyDeviceToHost);
        CUDA_CHECK(err);   
        err = cudaMemcpy(cpuY, gpuY, N * sizeof(float), cudaMemcpyDeviceToHost);
        CUDA_CHECK(err);   
        cudaDeviceSynchronize();
    }
    
    float getReferenceResults(int N) {
        double localValue = 0.0f;
        for (int i = 0; i<N; i++) {
            localValue += (double)cpuX[i] * (double)cpuY[i];
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
        float flops = N * 2.0f;
        float gflops = flops / (time_ms * 1e6f);
        return gflops;
    }
    
    /* Runner that can be called either by  the test handler or benchmark handler, and will return the average gflops */
    float runner(SdotImpl impl, std::vector<int> size, int nb_runs, int nb_warmup_runs, bool checkResult, bool printEach, int argc, char* argv[]) {
        float sum_gflops = 0.0f;
        allocate(size[0]);
        resetVectors(size[0]);
    
        for (int i = -1*nb_warmup_runs; i<nb_runs; i++) {
            float time_ms = impl(size[0], gpuX, gpuY, gpuOutput);
    
            
            if (checkResult) {
                compareToReference(size[0]);
            }
            
            if (i >= 0) {
    
                float gflops = getGflops(size[0], time_ms);
    
                if (printEach) {
                    printf("[internal][iter %d]: %f GFLOP/s \n", i, gflops);
                }
                sum_gflops += gflops;
            }
    
    
            resetVectors(size[0]);
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
    


}    




void sdot_handler(RunMode runMode, int argc, char * argv[]) {
    curandCreateGenerator(&sdot::gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(sdot::gen, 1234ULL);
    CUBLAS_CHECK(cublasCreate(&sdot::cublasHandle));

    switch (runMode) {
        case RunMode::TEST:
            Handlers<sdot::SdotImpl, 1, BENCH_RUNS, WARMUP_RUNS>::test_handler(argc, argv, sdot::sdotImpls, sdot::runner);
            break;
        case RunMode::BENCH:
            Handlers<sdot::SdotImpl, 1, BENCH_RUNS, WARMUP_RUNS>::bench_handler(argc, argv, sdot::sdotImpls, sdot::runner, sdot::SIZES, sdot::exportCSV);
            break;
    }

    CUBLAS_CHECK(cublasDestroy(sdot::cublasHandle));
    curandDestroyGenerator(sdot::gen);

}
