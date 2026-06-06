#pragma once

#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <cuda_runtime.h>
#include <time.h>
#include <unordered_map>
#include <map>
#include <vector>
#include <stdexcept>
#include <curand.h>
#include <curand_kernel.h>
#include "cublas_v2.h"
#include "enums.h"

#define LIB_HAS(name) LIB_HAS_##name

namespace sgemv {
    extern cublasHandle_t cublasHandle;
}

namespace sdot {
    extern cublasHandle_t cublasHandle;
}



#define CUDA_CHECK(error)                                                     \
    do {                                                                            \
        cudaError_t check_cuda_error_ = (error);                                    \
        if (check_cuda_error_ != cudaSuccess) {                                     \
            fprintf(stderr, "WARNING: CUDA error at %s:%d: %s\n", __FILE__,         \
                    __LINE__, cudaGetErrorString(check_cuda_error_));              \
        }                                                                           \
    } while (0)

#define CURAND_CHECK(call) do {                                 \
    curandStatus_t status = (call);                             \
    if (status != CURAND_STATUS_SUCCESS) {                      \
        fprintf(stderr, "cuRAND error: %d at %s:%d\n",         \
                (int)status, __FILE__, __LINE__);              \
        exit(EXIT_FAILURE);                                     \
    }                                                           \
} while(0)

#define CUBLAS_CHECK(call) do {                                 \
    cublasStatus_t status = (call);                             \
    if (status != CUBLAS_STATUS_SUCCESS) {                      \
        fprintf(stderr, "cuBLAS error: %d at %s:%d\n",         \
                (int)status, __FILE__, __LINE__);              \
        exit(EXIT_FAILURE);                                     \
    }                                                           \
} while(0)

void reduction_handler(RunMode runMode, int argc, char * argv[]);
void sdot_handler(RunMode runMode, int argc, char * argv[]);
void sgemv_handler(RunMode runMode, int argc, char * argv[]);


/* Common helpers */
float getRdmFloat(float lowerBound, float upperBound);


/* Implementation will either be ReductionImpl, SdotImpl, etc...
Size dims is the number of sizes, it could be 3 (a gemm for example, M, N, K), or any number*/
template <typename Implementation, int sizeDims, int BENCH_RUNS, int WARMUP_RUNS>
struct Handlers {
    using RunnerFunc = float (*)(Implementation, std::vector<int> sizes, int nb_runs, int nb_warmup_runs, bool checkResult, bool printEach, int argc, char * argv[]);
    
    static void test_handler(int argc, char * argv[], const std::unordered_map<std::string, Implementation>& implementationsMap, RunnerFunc runnerFunc) {
        /*
            ./libthom test <routine> <impl> <size> <nb_iter> <check_result=1>
        */
        int supposedArgc = 5 + sizeDims;
        if (argc < supposedArgc) {
            fprintf(stderr, "Missing arguments.\nUsage: ./libthom test <routine> <impl>");
            for (int i = 0; i<sizeDims; i++) {fprintf(stderr, " <size%d>", i+1);}
            fprintf(stderr, " <nb_iter> <check_result=1>\n");
            fprintf(stderr, "Available implementations:");
            for (const auto& entry : implementationsMap) {
                fprintf(stderr, " %s", entry.first.c_str());
            }
            fprintf(stderr, "\n");
            exit(1);
        }
        char * implName = argv[3];
        try {
            Implementation impl = implementationsMap.at(implName);
            std::vector<int> sizes;
            for (int i = 0; i<sizeDims; i++) {
                sizes.push_back(atoi(argv[i+4])); // sizes start at index 4
            }
            int nb_iter = atoi(argv[4 + sizeDims]);
            bool checkResult = true;
            if (argc == supposedArgc + 1) {checkResult = atoi(argv[4 + sizeDims + 1]) == 1;}
    
            float gflops = runnerFunc(impl, sizes, nb_iter, 0, checkResult, true, argc ,argv);
            printf("# GFLOP/s: %f\n", gflops);
    
        }
        catch (const std::out_of_range&) {
            fprintf(stderr, "Invalid implementation. Either it doesnt exist, or it's not available for the ARCH you've compiled\n");
            exit(1);
        }
    }

    static void bench_handler(int argc, char* argv[], const std::unordered_map<std::string, Implementation>& implementationsMap, RunnerFunc runnerFunc, const std::vector<std::vector<int>>& SIZES, void(*exportFunc)(const std::map<std::vector<int>,float>&, const std::string&)) {
        /*
            ./libthom bench <routine> <impl>
        */
        if (argc < 4) {
            fprintf(stderr, "Missing arguments.\nUsage: ./libthom bench <routine> <impl>\n");
            fprintf(stderr, "Available implementations:");
            for (const auto& entry : implementationsMap) {
                fprintf(stderr, " %s", entry.first.c_str());
            }
            fprintf(stderr, "\n");
            exit(1);
        }
        char * implName = argv[3];
        std::map<std::vector<int>, float> results;
    
        try {
            Implementation impl = implementationsMap.at(implName);
            for (std::vector<int> sizes : SIZES) {
                printf("\nStarting benchmark for size %d...\n", sizes[0]);
                float gflops = runnerFunc(impl, sizes, BENCH_RUNS, WARMUP_RUNS, false, true, argc, argv);
                results[sizes] = gflops;
                printf("[size: %d] ==> %f\n", sizes[0], gflops);
            }
            std::string filename = std::string("bench_") + implName + ".csv";
            exportFunc(results, filename);
        }
        catch (const std::out_of_range&) {
            fprintf(stderr, "Invalid implementation. Either it doesnt exist, or it's not available for the ARCH you've compiled\n");
            fprintf(stderr, "Available implementations:");
            for (const auto& entry : implementationsMap) {
                fprintf(stderr, " %s", entry.first.c_str());
            }
            fprintf(stderr, "\n");
            exit(1);
        }
    }
};
