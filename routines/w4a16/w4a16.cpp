#include "w4a16.h"
#include <cstddef>
#include <cmath>
#include <fstream>


#define BENCH_RUNS 20
#define WARMUP_RUNS 5

// The comparison to the reference prints at most this many errors
#define MAX_REPORTED_ERRORS 8



namespace w4a16 {
    /* M sweep at K = N = 8192: small M (LLM decoding) is memory bound, large M becomes compute bound */
    static const std::vector<std::vector<int>> SIZES = {
        {1, 8192, 8192},
        {2, 8192, 8192},
        {4, 8192, 8192},
        {8, 8192, 8192},
        {16, 8192, 8192},
        {32, 8192, 8192},
        {64, 8192, 8192},
        {128, 8192, 8192},
        {256, 8192, 8192}
    };

    static __half * cpuA;
    static __half * cpuW;               // original fp16 weights, only used to quantize
    static uint8_t * cpuW_int4;         // one int4 value per byte, before packing
    static uint32_t * cpuW_int4_packed;
    static __half * cpuScales;
    static __half * cpuW_dequant;
    static __half * cpuC_ref;
    static __half * resultFromGpu;
    static float * cpuRandom;           // host copy of the cuRAND buffer

    static float * gpuRandom;
    static __half * gpuA;
    static uint32_t * gpuW_int4_packed;
    static __half * gpuScales;
    static __half * gpuW_dequant;
    static __half * gpuC;
    static __half * gpuC_ref;


    cublasHandle_t cublasHandle;
    curandGenerator_t gen;
    static cudaError_t err;

    /* Handlers only pass one float per size to exportCSV (the GFLOP/s), so the runner stores everything else here */
    struct Metrics {
        float time_us;
        float gflops;
        float gbps;
        float pct_peak_bw;
        float cublas_time_us;
        float speedup_vs_cublas;
    };
    static std::map<std::vector<int>, Metrics> metrics;


    /* Registering implementations, only if they're compiled. They are compiled if we are using the correct Compute Capability*/
    static const std::unordered_map<std::string, W4A16Impl> w4a16Impls = {
    #if LIB_HAS(W4A16_CUBLAS)
        {"cublas", {w4a16_cublas, WeightFormat::FP16}},
    #endif
    #if LIB_HAS(W4A16_NAIVE)
        {"naive", {w4a16_naive, WeightFormat::INT4}},
    #endif
    };



    void allocate(int M, int N, int K) {
        cpuA = (__half*)malloc((size_t)M * K * sizeof(__half));
        if (cpuA == NULL) {fprintf(stderr, "Error allocating cpuA\n");}
        cpuW = (__half*)malloc((size_t)K * N * sizeof(__half));
        if (cpuW == NULL) {fprintf(stderr, "Error allocating cpuW\n");}
        cpuW_int4 = (uint8_t*)malloc((size_t)K * N * sizeof(uint8_t));
        if (cpuW_int4 == NULL) {fprintf(stderr, "Error allocating cpuW_int4\n");}
        cpuW_int4_packed = (uint32_t*)malloc((size_t)(K / INT4_PER_UINT32) * N * sizeof(uint32_t));
        if (cpuW_int4_packed == NULL) {fprintf(stderr, "Error allocating cpuW_int4_packed\n");}
        cpuScales = (__half*)malloc((size_t)(K / GROUP_SIZE) * N * sizeof(__half));
        if (cpuScales == NULL) {fprintf(stderr, "Error allocating cpuScales\n");}
        cpuW_dequant = (__half*)malloc((size_t)K * N * sizeof(__half));
        if (cpuW_dequant == NULL) {fprintf(stderr, "Error allocating cpuW_dequant\n");}
        cpuC_ref = (__half*)malloc((size_t)M * N * sizeof(__half));
        if (cpuC_ref == NULL) {fprintf(stderr, "Error allocating cpuC_ref\n");}
        resultFromGpu = (__half*)malloc((size_t)M * N * sizeof(__half));
        if (resultFromGpu == NULL) {fprintf(stderr, "Error allocating resultFromGpu\n");}

        // the random buffer is shared by A and W, so it has the size of the biggest one
        size_t randomSize = std::max((size_t)M * K, (size_t)K * N);
        cpuRandom = (float*)malloc(randomSize * sizeof(float));
        if (cpuRandom == NULL) {fprintf(stderr, "Error allocating cpuRandom\n");}

        err = cudaMalloc((void **)&gpuRandom, randomSize * sizeof(float));
        CUDA_CHECK(err);
        err = cudaMalloc((void **)&gpuA, (size_t)M * K * sizeof(__half));
        CUDA_CHECK(err);
        err = cudaMalloc((void **)&gpuW_int4_packed, (size_t)(K / INT4_PER_UINT32) * N * sizeof(uint32_t));
        CUDA_CHECK(err);
        err = cudaMalloc((void **)&gpuScales, (size_t)(K / GROUP_SIZE) * N * sizeof(__half));
        CUDA_CHECK(err);
        err = cudaMalloc((void **)&gpuW_dequant, (size_t)K * N * sizeof(__half));
        CUDA_CHECK(err);
        err = cudaMalloc((void **)&gpuC, (size_t)M * N * sizeof(__half));
        CUDA_CHECK(err);
        err = cudaMalloc((void **)&gpuC_ref, (size_t)M * N * sizeof(__half));
        CUDA_CHECK(err);
    }

    void freeAll() {
        free(cpuA);
        free(cpuW);
        free(cpuW_int4);
        free(cpuW_int4_packed);
        free(cpuScales);
        free(cpuW_dequant);
        free(cpuC_ref);
        free(resultFromGpu);
        free(cpuRandom);
        cudaFree(gpuRandom);
        cudaFree(gpuA);
        cudaFree(gpuW_int4_packed);
        cudaFree(gpuScales);
        cudaFree(gpuW_dequant);
        cudaFree(gpuC);
        cudaFree(gpuC_ref);
    }


    W4A16Problem getProblem(int M, int N, int K) {
        return {gpuA, gpuW_int4_packed, gpuScales, gpuW_dequant, gpuC, (size_t)M, (size_t)N, (size_t)K};
    }


    /* Fill dst (host) with fp16 values uniformly distributed in [-1, 1], generated with cuRAND on the GPU.
    Weights need to be signed for the symmetric quantization to make sense */
    void generateHalf(__half * dst, size_t size) {
        CURAND_CHECK(curandGenerateUniform(gen, gpuRandom, size));
        err = cudaMemcpy(cpuRandom, gpuRandom, size * sizeof(float), cudaMemcpyDeviceToHost);
        CUDA_CHECK(err);
        for (size_t i = 0; i < size; i++) {
            dst[i] = __float2half(2.0f * cpuRandom[i] - 1.0f);  // curandGenerateUniform returns values in (0, 1]
        }
    }


    /* Done once per size: A and W are never modified by the implementations (C is overwritten, beta = 0),
    so unlike other routines we don't regenerate them between runs */
    void initProblem(int M, int N, int K) {
        // 1 - random A and W
        generateHalf(cpuA, (size_t)M * K);
        generateHalf(cpuW, (size_t)K * N);

        // 2 - offline quantization on the host
        quantize_w4(cpuW, cpuW_int4, cpuScales, K, N);
        pack_w4(cpuW_int4, cpuW_int4_packed, K, N);
        dequantize_w4(cpuW_int4, cpuScales, cpuW_dequant, K, N);

        // 3 - copy everything the implementations may read
        err = cudaMemcpy(gpuA, cpuA, (size_t)M * K * sizeof(__half), cudaMemcpyHostToDevice);
        CUDA_CHECK(err);
        err = cudaMemcpy(gpuW_int4_packed, cpuW_int4_packed, (size_t)(K / INT4_PER_UINT32) * N * sizeof(uint32_t), cudaMemcpyHostToDevice);
        CUDA_CHECK(err);
        err = cudaMemcpy(gpuScales, cpuScales, (size_t)(K / GROUP_SIZE) * N * sizeof(__half), cudaMemcpyHostToDevice);
        CUDA_CHECK(err);
        // only for the cuBLAS reference, my kernel is not allowed to read that it would be cheating
        err = cudaMemcpy(gpuW_dequant, cpuW_dequant, (size_t)K * N * sizeof(__half), cudaMemcpyHostToDevice);
        CUDA_CHECK(err);

        // 4 - reference result C_ref = A * W_dequant, computed once with cuBLAS
        W4A16Problem referenceProblem = getProblem(M, N, K);
        referenceProblem.C = gpuC_ref;
        w4a16_cublas(referenceProblem);
        CUDA_CHECK(cudaDeviceSynchronize());
        err = cudaMemcpy(cpuC_ref, gpuC_ref, (size_t)M * N * sizeof(__half), cudaMemcpyDeviceToHost);
        CUDA_CHECK(err);
    }


    /* An element is wrong if its error is above 1% of its value plus an absolute margin.
    The margin is relative to the typical magnitude of C (its RMS): each element is a sum of K products, so small
    rounding differences (e.g. a kernel that doesn't round the dequantized weights to fp16 like the reference)
    add up to an absolute error that grows with K. On elements close to 0 (terms cancelling out) that error is
    a big relative error, without being a bug. A real bug (wrong index, wrong scale) gives errors of the order of
    the RMS itself, far above the margin */
    bool isClose(float result, float reference, float absoluteTolerance) {
        float diff = std::fabs(result - reference);
        return diff <= 1e-2f * std::fabs(reference) + absoluteTolerance;
    }

    float getRms(const __half * values, size_t size) {
        double sumSquares = 0.0;
        for (size_t i = 0; i < size; i++) {
            double v = __half2float(values[i]);
            sumSquares += v * v;
        }
        return std::sqrt(sumSquares / size);
    }


    bool compareToReference(int M, int N) {
        err = cudaMemcpy(resultFromGpu, gpuC, (size_t)M * N * sizeof(__half), cudaMemcpyDeviceToHost);
        CUDA_CHECK(err);

        const float absoluteTolerance = 1e-3f * getRms(cpuC_ref, (size_t)M * N);

        int countDifferent = 0;
        for (int row = 0; row < M; row++) {
            for (int col = 0; col < N; col++) {
                float result = __half2float(resultFromGpu[(size_t)row * N + col]);
                float reference = __half2float(cpuC_ref[(size_t)row * N + col]);
                if (!isClose(result, reference, absoluteTolerance)) {
                    countDifferent += 1;
                    if (countDifferent <= MAX_REPORTED_ERRORS) {
                        printf("row %d col %d | GPU: %f | reference: %f\n", row, col, result, reference);
                    }
                }
            }
        }

        if (countDifferent > 0) {
            printf("Number of different results: %d (tolerance: 1%% + %f)\n", countDifferent, absoluteTolerance);
        }
        return countDifferent == 0;
    }


    /* Time nb_runs calls of an implementation (after nb_warmup_runs untimed ones), flushing the L2 before each call */
    std::vector<float> measure(W4A16Function function, int M, int N, int K, int nb_runs, int nb_warmup_runs, bool checkResult) {
        std::vector<float> times_ms;
        W4A16Problem problem = getProblem(M, N, K);

        for (int i = -1 * nb_warmup_runs; i < nb_runs; i++) {
            if (checkResult) {
                // so that a kernel writing nothing can't pass thanks to the previous run's result
                CUDA_CHECK(cudaMemset(gpuC, 0, (size_t)M * N * sizeof(__half)));
            }
            flushL2();

            float time_ms = function(problem);
            CUDA_CHECK(cudaGetLastError());

            if (checkResult) {
                compareToReference(M, N);
            }
            if (i >= 0) {
                times_ms.push_back(time_ms);
            }
        }
        return times_ms;
    }

    float median(std::vector<float> values) {
        std::sort(values.begin(), values.end());
        size_t middle = values.size() / 2;
        return values.size() % 2 == 1 ? values[middle] : 0.5f * (values[middle - 1] + values[middle]);
    }

    /* Minimal number of bytes an implementation has to move between DRAM and the SMs */
    double getBytesMoved(WeightFormat weights, int M, int N, int K) {
        double bytesA = (double)M * K * sizeof(__half);
        double bytesC = (double)M * N * sizeof(__half);
        double bytesW;
        if (weights == WeightFormat::FP16) {
            bytesW = (double)K * N * sizeof(__half);
        } else {
            bytesW = (double)(K / INT4_PER_UINT32) * N * sizeof(uint32_t)       // packed int4 weights
                   + (double)(K / GROUP_SIZE) * N * sizeof(__half);             // scales
        }
        return bytesA + bytesW + bytesC;
    }

    float getGflops(int M, int N, int K, float time_ms) {
        double flops = 2.0 * M * N * K;
        return flops / (time_ms * 1e6);
    }


    /* Runner that can be called either by the test handler or benchmark handler, and will return the median GFLOP/s.
    It also times the cuBLAS fp16 baseline on the same data to get the speedup, both use the median of their times */
    float runner(W4A16Impl impl, std::vector<int> sizes, int nb_runs, int nb_warmup_runs, bool checkResult, bool printEach, int argc, char * argv[]) {
        int M = sizes[0];
        int N = sizes[1];
        int K = sizes[2];

        if (K % GROUP_SIZE != 0) {
            fprintf(stderr, "K must be a multiple of the group size (%d)\n", GROUP_SIZE);
            exit(1);
        }
        nb_runs = std::max(nb_runs, 1);
        // test mode passes 0 warmup runs, but the first cuBLAS call includes its initialization
        nb_warmup_runs = std::max(nb_warmup_runs, 1);

        allocate(M, N, K);
        initProblem(M, N, K);

        std::vector<float> times_ms = measure(impl.function, M, N, K, nb_runs, nb_warmup_runs, checkResult);
        std::vector<float> cublas_times_ms = measure(w4a16_cublas, M, N, K, nb_runs, nb_warmup_runs, false);

        if (printEach) {
            for (int i = 0; i < nb_runs; i++) {
                printf("[internal][iter %d]: %f GFLOP/s | %.2f us (cuBLAS fp16: %.2f us)\n",
                    i, getGflops(M, N, K, times_ms[i]), times_ms[i] * 1e3f, cublas_times_ms[i] * 1e3f);
            }
        }

        static const float peakBandwidth = getPeakBandwidthGBs();
        float time_ms = median(times_ms);
        float cublas_time_ms = median(cublas_times_ms);

        Metrics result;
        result.time_us = time_ms * 1e3f;
        result.gflops = getGflops(M, N, K, time_ms);
        result.gbps = getBytesMoved(impl.weights, M, N, K) / (time_ms * 1e6);
        result.pct_peak_bw = 100.0f * result.gbps / peakBandwidth;
        result.cublas_time_us = cublas_time_ms * 1e3f;
        result.speedup_vs_cublas = cublas_time_ms / time_ms;
        metrics[sizes] = result;

        printf("# median: %.2f us | %.1f GB/s (%.1f%% of %.0f GB/s peak) | cuBLAS fp16: %.2f us | speedup vs cuBLAS: %.2fx\n",
            result.time_us, result.gbps, result.pct_peak_bw, peakBandwidth, result.cublas_time_us, result.speedup_vs_cublas);

        freeAll();

        return result.gflops;
    }

    /* BENCHMARKING */

    void exportCSV(const std::map<std::vector<int>, float>& results,
                    const std::string& filename)
    {
        std::ofstream file(filename);

        file << "M,N,K,time_us,gflops,gbps,pct_peak_bw,cublas_time_us,speedup_vs_cublas\n";

        for (const auto& [sizes, gflops] : results) {
            const Metrics& m = metrics.at(sizes);
            file << sizes[0] << "," << sizes[1] << "," << sizes[2] << ","
                 << m.time_us << "," << m.gflops << "," << m.gbps << "," << m.pct_peak_bw << ","
                 << m.cublas_time_us << "," << m.speedup_vs_cublas << "\n";
        }
    }

}




void w4a16_handler(RunMode runMode, int argc, char * argv[]) {
    CURAND_CHECK(curandCreateGenerator(&w4a16::gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(w4a16::gen, 1234ULL));
    CUBLAS_CHECK(cublasCreate(&w4a16::cublasHandle));

    switch (runMode) {
        case RunMode::TEST:
            Handlers<W4A16Impl, 3, BENCH_RUNS, WARMUP_RUNS>::test_handler(argc, argv, w4a16::w4a16Impls, w4a16::runner);
            break;
        case RunMode::BENCH:
            Handlers<W4A16Impl, 3, BENCH_RUNS, WARMUP_RUNS>::bench_handler(argc, argv, w4a16::w4a16Impls, w4a16::runner, w4a16::SIZES, w4a16::exportCSV);
            break;
        default:
            fprintf(stderr, "Mode not supported by w4a16\n");
            break;
    }

    CURAND_CHECK(curandDestroyGenerator(w4a16::gen));
    CUBLAS_CHECK(cublasDestroy(w4a16::cublasHandle));
}
