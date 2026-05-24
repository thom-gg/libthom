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

void reduction_handler(RunMode runMode, int argc, char * argv[]);
void scan_handler(RunMode runMode, int argc, char * argv[]);


/* Common helpers */
double getRdmDouble(double lowerBound, double upperBound);