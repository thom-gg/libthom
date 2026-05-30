#pragma once

#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <cuda_runtime.h>
#include <time.h>
#include <unordered_map>
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