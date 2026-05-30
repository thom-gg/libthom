#pragma once

#include "../routines.h"

#if LIB_HAS(REDUCTION_WARP)
float reduction_warp(const float* input,  float* output, int N);
#endif

#if LIB_HAS(REDUCTION_THRUST)
float reduction_thrust(const float* input,  float* output, int N);
#endif