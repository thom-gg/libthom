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
#include "enums.h"

#define LIB_HAS(name) LIB_HAS_##name


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

void reduction_handler(RunMode runMode, int argc, char * argv[]);
void scan_handler(RunMode runMode, int argc, char * argv[]);


/* Common helpers */
float getRdmFloat(float lowerBound, float upperBound);


/* Implementation will either be ReductionImpl, SdotImpl, etc...*/
template <typename Implementation, int BENCH_RUNS, int WARMUP_RUNS>
struct Handlers {
    using RunnerFunc = float (*)(Implementation, int size, int nb_runs, int nb_warmup_runs, bool checkResult, bool printEach);
    
    static void test_handler(int argc, char * argv[], const std::unordered_map<std::string, Implementation>& implementationsMap, RunnerFunc runnerFunc) {
        /*
            ./libthom test <routine> <impl> <size> <nb_iter> <check_result=1>
        */
        if (argc != 6 && argc != 7) {
            fprintf(stderr, "Missing arguments.\nUsage: ./libthom test <routine> <impl> <size> <nb_iter> <check_result=1>\n");
            exit(1);
        }
        char * implName = argv[3];
        try {
            Implementation impl = implementationsMap.at(implName);
            int size = atoi(argv[4]);
            int nb_iter = atoi(argv[5]);
            bool checkResult = true;
            if (argc == 7) {checkResult = atoi(argv[6]) == 1;}
    
            
            float gflops = runnerFunc(impl, size, nb_iter, 0, checkResult, true);
            printf("# GFLOP/s: %f\n", gflops);
    
        }
        catch (const std::out_of_range&) {
            fprintf(stderr, "Invalid implementation. Either it doesnt exist, or it's not available for the ARCH you've compiled\n");
            exit(1);
        }
    }

    static void bench_handler(int argc, char* argv[], const std::unordered_map<std::string, Implementation>& implementationsMap, RunnerFunc runnerFunc, const std::vector<int>& SIZES, void(*exportFunc)(const std::map<int,float>&, const std::string&)) {
        /*
            ./libthom bench <routine> <impl>
        */
        if (argc != 4) {
            fprintf(stderr, "Missing arguments.\nUsage: ./libthom bench <routine> <impl>\n");
            exit(1);
        }
        char * implName = argv[3];
        std::map<int, float> results;
    
        try {
            Implementation impl = implementationsMap.at(implName);
            for (int size : SIZES) {
                printf("\nStarting benchmark for size %d...\n", size);
                float gflops = runnerFunc(impl, size, BENCH_RUNS, WARMUP_RUNS, false, true);
                results[size] = gflops;
                printf("[size: %d] ==> %f\n", size, gflops);
            }
            std::string filename = std::string("bench_") + implName + ".csv";
            exportFunc(results, filename);
        }
        catch (const std::out_of_range&) {
            fprintf(stderr, "Invalid implementation. Either it doesnt exist, or it's not available for the ARCH you've compiled\n");
            exit(1);
        }
    }
};
