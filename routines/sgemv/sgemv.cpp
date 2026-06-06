#include "sgemv.h"
#include <cstddef>
#include <fstream>
#include <nvtx3/nvToolsExt.h>
#include <curand.h>
#include <curand_kernel.h>

#define BENCH_RUNS 10
#define WARMUP_RUNS 5


#define LOWER_BOUND -5.0
#define UPPER_BOUND 5.0



namespace sgemv {
    static const std::vector<std::vector<int>> SIZES = {{1000,1000}};
    
    static float * cpuA;
    static float * cpuX;
    static float * cpuY;
    static float cpuAlpha;
    static float cpuBeta;
    static float * resultFromGpu;
    
    static float * gpuA;
    static float * gpuX;
    static float * gpuY;
    static float * gpuAlpha;
    static float * gpuBeta;
    
    
    cublasHandle_t cublasHandle;
    curandGenerator_t gen;
    static cudaError_t err;
    
    using SgemvImpl = float (*)(int m, int n, const float * alpha, const float * A, int lda, const float * x, int incx, const float * beta, float * y, int incy, bool rowMajor);
    
    
    /* Registering implementations, only if they're compiled. They are compiled if we are using the correct Compute Capability*/
    static const std::unordered_map<std::string, SgemvImpl> sgemvImpls = {
    #if LIB_HAS(SGEMV_NAIVE)
        {"sgemv_naive", sgemv_naive},
    #endif
    #if LIB_HAS(SGEMV_BLOCK_ROW)
        {"sgemv_block_row", sgemv_block_row},
    #endif
    #if LIB_HAS(SGEMV_COL1_NAIVE)
        {"sgemv_col1_naive", sgemv_col1_naive},
    #endif
    #if LIB_HAS(SGEMV_COL2_SHMEM)
        {"sgemv_col2_shmem", sgemv_col2_shmem},
    #endif
    #if LIB_HAS(SGEMV_COL3_VECTO)
        {"sgemv_col3_vecto", sgemv_col3_vecto},
    #endif
    #if LIB_HAS(SGEMV_COL4_TRANS)
        {"sgemv_col4_trans", sgemv_col4_trans},
    #endif
    #if LIB_HAS(SGEMV_CUBLAS)
        {"sgemv_cublas", sgemv_cublas},
    #endif
    #if LIB_HAS(SGEMV_VECTO)
        {"sgemv_vecto", sgemv_vecto},
    #endif
    };
 

    
    void allocate(int M, int N) {
        cpuA = (float*)malloc(M * N * sizeof(float));
        if (cpuA == NULL) {fprintf(stderr, "Error allocating cpuA\n");}
        cpuX = (float*)malloc(N * sizeof(float));
        if (cpuX == NULL) {fprintf(stderr, "Error allocating cpuX\n");}
        cpuY = (float*)malloc(M * sizeof(float));
        if (cpuY == NULL) {fprintf(stderr, "Error allocating cpuY\n");}
        resultFromGpu = (float*)malloc(M * sizeof(float));
        if (resultFromGpu == NULL) {fprintf(stderr, "Error allocating resultFromGpu\n");}
    
        err = cudaMalloc((void **)&gpuA, M*N*sizeof(float));
        CUDA_CHECK(err);
        err = cudaMalloc((void **)&gpuX, N*sizeof(float));
        CUDA_CHECK(err);
        err = cudaMalloc((void **)&gpuY, M*sizeof(float));
        CUDA_CHECK(err);
        err = cudaMalloc((void **)&gpuAlpha, 1*sizeof(float));
        CUDA_CHECK(err);
        err = cudaMalloc((void **)&gpuBeta, 1*sizeof(float));
        CUDA_CHECK(err);
    }
    
    void freeAll() {
        free(cpuA);
        free(cpuX);
        free(cpuY);
        free(resultFromGpu);
        cudaFree(gpuA);
        cudaFree(gpuX);
        cudaFree(gpuY);
        cudaFree(gpuAlpha);
        cudaFree(gpuBeta);
    }
    
    
    void resetVectors(int M, int N) {
        // fill randomly using cuRAND directly on gpu
        CURAND_CHECK(curandGenerateUniform(gen, gpuA, M*N));
        CURAND_CHECK(curandGenerateUniform(gen, gpuX, N));
        CURAND_CHECK(curandGenerateUniform(gen, gpuY, M));

        cpuAlpha = 1.0f;
        err = cudaMemcpy(gpuAlpha, &cpuAlpha, sizeof(float), cudaMemcpyHostToDevice);
        CUDA_CHECK(err);

        cpuBeta = 1.0f;
        err = cudaMemcpy(gpuBeta, &cpuBeta, sizeof(float), cudaMemcpyHostToDevice);
        CUDA_CHECK(err);
        cudaDeviceSynchronize();

        // copy on cpu for reference computations
        err = cudaMemcpy(cpuX, gpuX, N * sizeof(float), cudaMemcpyDeviceToHost);
        CUDA_CHECK(err);   
        err = cudaMemcpy(cpuA, gpuA, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        CUDA_CHECK(err); 
        err = cudaMemcpy(cpuY, gpuY, M * sizeof(float), cudaMemcpyDeviceToHost);
        CUDA_CHECK(err);   
        cudaDeviceSynchronize();
    }
    
    // after this call cpuY will contain results from the reference implementation (CPU)
    void getReferenceResults(int M, int N, bool rowMajor, int lda) {
        int incx = 1;
        int incy = 1;
        for (int row = 0; row < M; row++) {
            double dot = 0.0;
            for (int col = 0; col < N; col++) {
                if (rowMajor) {
                    dot += (double)cpuA[row * lda + col] * (double)cpuX[col * incx];
                }
                else {
                    dot += (double)cpuA[col * lda + row] * (double)cpuX[col * incx];
                }
            }
            cpuY[row * incy] = (float)(cpuAlpha * dot + cpuBeta * (double)cpuY[row * incy]);
        }
    }

    
    void compareToReference(int M,int N, bool rowMajor, int lda) {
        err = cudaMemcpy(resultFromGpu, gpuY, M*sizeof(float), cudaMemcpyDeviceToHost);
        CUDA_CHECK(err);
        getReferenceResults(M,N,rowMajor, lda);

        int countDifferent = 0;
        for (int i = 0; i<M; i++) {
            float rel_error = fabsf(resultFromGpu[i] -  cpuY[i]) / fabsf( cpuY[i]);
            if (rel_error > 1e-3f) {
                countDifferent++;
                // printf("Different results at index %d: relative error: %f\n", i, rel_error);
            }
        }
        if (countDifferent > 0) {
            printf("Number of different results: %d\n", countDifferent);
        }         
    }
    
    float getGflops(int M, int N, float time_ms) {
        long long flops = 2LL * M * N;
        float time_s = time_ms / 1000.0f;
        return (float)(flops / time_s) / 1e9f;
    }
    
    /* Runner that can be called either by  the test handler or benchmark handler, and will return the average gflops */
    float runner(SgemvImpl impl, std::vector<int> sizes, int nb_runs, int nb_warmup_runs, bool checkResult, bool printEach, int argc, char * argv[]) {
        int M = sizes[0];
        int N = sizes[1];
        /* row major parameter can be passed as a last parameter with value (row or column)
        this information of an additional parameter should be passed to the test / bench handlers so that it can didsplay a correct usage string*/
        bool rowMajor = strcmp(argv[argc-1], "row") == 0;
        int lda = rowMajor ? N : M;
        float sum_gflops = 0.0f;
        allocate(M, N);
        resetVectors(M, N);

        for (int i = -1*nb_warmup_runs; i<nb_runs; i++) {
            float time_ms = impl(M, N, gpuAlpha, gpuA, lda, gpuX, 1, gpuBeta, gpuY, 1, rowMajor);
    
            
            if (checkResult) {
                compareToReference(M, N, rowMajor, lda);
            }
            
            if (i >= 0) {
    
                float gflops = getGflops(M, N, time_ms);
    
                if (printEach) {
                    printf("[internal][iter %d]: %f GFLOP/s \n", i, gflops);
                }
                sum_gflops += gflops;
            }
    
    
            resetVectors(M, N);
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
    
        file << "M,N,gflops\n";
    
        for (const auto& [sizes, gflops] : results) {
            file << sizes[0] << "," << sizes[1] << "," << gflops << "\n";
        }
    }
    


}    




void sgemv_handler(RunMode runMode, int argc, char * argv[]) {
    CURAND_CHECK(curandCreateGenerator(&sgemv::gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(sgemv::gen, 1234ULL));
    CUBLAS_CHECK(cublasCreate(&sgemv::cublasHandle));

    switch (runMode) {
        case RunMode::TEST:
            Handlers<sgemv::SgemvImpl, 2, BENCH_RUNS, WARMUP_RUNS>::test_handler(argc, argv, sgemv::sgemvImpls, sgemv::runner);
            break;
        case RunMode::BENCH:
            Handlers<sgemv::SgemvImpl, 2, BENCH_RUNS, WARMUP_RUNS>::bench_handler(argc, argv, sgemv::sgemvImpls, sgemv::runner, sgemv::SIZES, sgemv::exportCSV);
            break;
    }

    CURAND_CHECK(curandDestroyGenerator(sgemv::gen));
    CUBLAS_CHECK(cublasDestroy(sgemv::cublasHandle));

}
