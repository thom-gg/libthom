#include "routines.h"


float getRdmFloat(float lowerBound, float upperBound) {
    return lowerBound + ((float)rand() / RAND_MAX) * (upperBound-lowerBound);
}


/* Flush the L2 cache by writing a buffer twice its size, so the next kernel reads its inputs from DRAM.
Without it, inputs smaller than the L2 (e.g. a 32 MB int4 weight matrix on a 40 MB A100 L2) stay cached
between runs and the benchmark measures L2 bandwidth instead of DRAM bandwidth. */
void flushL2() {
    static void * flushBuffer = nullptr;
    static size_t flushBufferSize = 0;

    if (flushBuffer == nullptr) {
        int device, l2Size;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaDeviceGetAttribute(&l2Size, cudaDevAttrL2CacheSize, device));
        flushBufferSize = 2 * (size_t)l2Size;
        CUDA_CHECK(cudaMalloc(&flushBuffer, flushBufferSize));
    }
    CUDA_CHECK(cudaMemset(flushBuffer, 0, flushBufferSize));
}

/* Theoretical peak DRAM bandwidth of the current device in GB/s:
2 transfers per clock (DDR) * memory clock * bus width in bytes */
float getPeakBandwidthGBs() {
    int device, memoryClockKHz, busWidthBits;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&memoryClockKHz, cudaDevAttrMemoryClockRate, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&busWidthBits, cudaDevAttrGlobalMemoryBusWidth, device));
    return 2.0f * memoryClockKHz * 1e3f * (busWidthBits / 8.0f) / 1e9f;
}
