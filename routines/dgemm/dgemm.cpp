#include "dgemm.h"
#include <cstddef>
#include <fstream>
#include <nvtx3/nvToolsExt.h>


#define BENCH_RUNS 10
#define WARMUP_RUNS 5


#define LOWER_BOUND -5.0
#define UPPER_BOUND 5.0



namespace dgemm {
    static const std::vector<std::vector<int>> SIZES = {{512,512,512}, {1024,1024,1024}, {2048, 2048, 2048}};
    
    static double * cpuA;
    static double * cpuB;
    static double * cpuC;
    static double cpuAlpha;
    static double cpuBeta;
    static double * resultFromGpu;
    
    static double * gpuA;
    static double * gpuB;
    static double * gpuC;

    
    
    cublasHandle_t cublasHandle;
    curandGenerator_t gen;
    static cudaError_t err;
    
    using DgemmImpl = float (*)(const double* A, const double* B, double* C, size_t M, size_t N, size_t K,double alpha, double beta);
    
    
    /* Registering implementations, only if they're compiled. They are compiled if we are using the correct Compute Capability*/
    static const std::unordered_map<std::string, DgemmImpl> dgemmImpls = {
    #if LIB_HAS(DGEMM_CUBLAS)
        {"cublas", dgemm_cublas},
    #endif
    #if LIB_HAS(DGEMM_BLOCKTILING)
        {"blocktiling", dgemm_blocktiling},
    #endif
    #if LIB_HAS(DGEMM_CPASYNC)
        {"cpasync", dgemm_cpasync},
    #endif
    #if LIB_HAS(DGEMM_CUBLAS_TC)
        {"cublas_tc", dgemm_cublas_tc},
    #endif
    #if LIB_HAS(DGEMM_FINAL)
        {"final", dgemm_final},
    #endif
    #if LIB_HAS(DGEMM_SCHOOL)
        {"school", dgemm_school},
    #endif
    #if LIB_HAS(DGEMM_SHMEM)
        {"shmem", dgemm_shmem},
    #endif
    #if LIB_HAS(DGEMM_TENSORCORES)
        {"tensorcores", dgemm_tensorcores},
    #endif
    #if LIB_HAS(DGEMM_TRIPLEBUF)
        {"triplebuf", dgemm_triplebuf},
    #endif
    #if LIB_HAS(DGEMM_WARPTILING)
        {"warptiling", dgemm_warptiling},
    #endif
    };
 

    
    void allocate(int M, int N, int K) {
        cpuA = (double*)malloc(M * K * sizeof(double));
        if (cpuA == NULL) {fprintf(stderr, "Error allocating cpuA\n");}
        cpuB = (double*)malloc(K * N * sizeof(double));
        if (cpuB == NULL) {fprintf(stderr, "Error allocating cpuB\n");}
        cpuC = (double*)malloc(M * N * sizeof(double));
        if (cpuC == NULL) {fprintf(stderr, "Error allocating cpuC\n");}
        resultFromGpu = (double*) malloc(M*N * sizeof(double));
        if (resultFromGpu == NULL) {fprintf(stderr,"Error allocating resultFormGpu\n");}
    
        err = cudaMalloc((void **)&gpuA, M*K*sizeof(double));
        CUDA_CHECK(err);
        err = cudaMalloc((void **)&gpuB, K*N*sizeof(double));
        CUDA_CHECK(err);
        err = cudaMalloc((void **)&gpuC, M*N*sizeof(double));
        CUDA_CHECK(err);
    }
    
    void freeAll() {
        free(cpuA);
        free(cpuB);
        free(cpuC);
        cudaFree(gpuA);
        cudaFree(gpuB);
        cudaFree(gpuC);
    }
    
    
    void resetVectors(int M, int N, int K) {
        // fill randomly using cuRAND directly on gpu
        CURAND_CHECK(curandGenerateUniformDouble(gen, gpuA, M*K));
        CURAND_CHECK(curandGenerateUniformDouble(gen, gpuB, K*N));
        CURAND_CHECK(curandGenerateUniformDouble(gen, gpuC, M*N));

        cpuAlpha = 1.0;
        cpuBeta = 1.0;

        cudaDeviceSynchronize();

        // copy on cpu for reference computations
        err = cudaMemcpy(cpuA, gpuA, M*K* sizeof(double), cudaMemcpyDeviceToHost);
        CUDA_CHECK(err);   
        err = cudaMemcpy(cpuB, gpuB, K * N * sizeof(double), cudaMemcpyDeviceToHost);
        CUDA_CHECK(err); 
        err = cudaMemcpy(cpuC, gpuC, M * N * sizeof(double), cudaMemcpyDeviceToHost);
        CUDA_CHECK(err);   
        cudaDeviceSynchronize();
    }
    
    // after this call cpuC will contain results from the reference implementation (CPU)
    void getReferenceResults(int M, int N, int K) {
        for (int row= 0; row<M; row++) {
            for (int col = 0; col<N; col++) {
                // inner dot product
                double sum = 0;
                for (int k = 0; k<K; k++) {
                    sum += cpuA[row * K + k] * cpuB[k * N + col];
                }
                cpuC[row * N + col] = cpuAlpha * sum + cpuBeta * cpuC[row * N + col];
            }
        }
    }


    
    void compareToReference(int M,int N, int K) {
        err = cudaMemcpy(resultFromGpu, gpuC, M*N*sizeof(double), cudaMemcpyDeviceToHost);
        CUDA_CHECK(err);
        getReferenceResults(M,N,K);

        int countDifferent = 0;
        for (int row= 0; row<M; row++) {
            for (int col = 0; col<N; col++) {
                double diff = fabsf(resultFromGpu[row*N+col] - cpuC[row*N+col]);
                double abs_ref = fabsf(cpuC[row*N+col]);
                double rel_error = diff / abs_ref;
                if (rel_error > 1e-3f) {
                    countDifferent+=1;
                    printf("row %d col %d | GPU: %f | CPU: %f | rel_error: %f\n",
                        row, col, resultFromGpu[row*N+col], cpuC[row*N+col], rel_error);
                }
            }
        }

        if (countDifferent > 0) {
            printf("Number of different results: %d\n", countDifferent);
        }         
    }
    
    float getGflops(int M, int N, int K, float time_ms) {
        float flops = 2.0 * M * N * K;
        float gflops = flops / (time_ms * 1e6f);
        return gflops;
    }
    
    /* Runner that can be called either by  the test handler or benchmark handler, and will return the average gflops */
    float runner(DgemmImpl impl, std::vector<int> sizes, int nb_runs, int nb_warmup_runs, bool checkResult, bool printEach, int argc, char * argv[]) {
        int M = sizes[0];
        int N = sizes[1];
        int K = sizes[2];
        
        float sum_gflops = 0.0f;
        allocate(M, N, K);
        resetVectors(M, N, K);

        for (int i = -1*nb_warmup_runs; i<nb_runs; i++) {
            float time_ms = impl(gpuA, gpuB, gpuC, M, N, K, cpuAlpha, cpuBeta);
    
            
            if (checkResult) {
                compareToReference(M, N, K);
            }
            
            if (i >= 0) {
    
                float gflops = getGflops(M, N, K, time_ms);
    
                if (printEach) {
                    printf("[internal][iter %d]: %f GFLOP/s \n", i, gflops);
                }
                sum_gflops += gflops;
            }
    
    
            resetVectors(M, N, K);
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
    
        file << "M,N,K,gflops\n";
    
        for (const auto& [sizes, gflops] : results) {
            file << sizes[0] << "," << sizes[1] << "," << sizes[2] << "," << gflops << "\n";
        }
    }
    


}    




void dgemm_handler(RunMode runMode, int argc, char * argv[]) {
    CURAND_CHECK(curandCreateGenerator(&dgemm::gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(dgemm::gen, 1234ULL));
    CUBLAS_CHECK(cublasCreate(&dgemm::cublasHandle));

    switch (runMode) {
        case RunMode::TEST:
            Handlers<dgemm::DgemmImpl, 3, BENCH_RUNS, WARMUP_RUNS>::test_handler(argc, argv, dgemm::dgemmImpls, dgemm::runner);
            break;
        case RunMode::BENCH:
            Handlers<dgemm::DgemmImpl, 3, BENCH_RUNS, WARMUP_RUNS>::bench_handler(argc, argv, dgemm::dgemmImpls, dgemm::runner, dgemm::SIZES, dgemm::exportCSV);
            break;
    }

    CURAND_CHECK(curandDestroyGenerator(dgemm::gen));
    CUBLAS_CHECK(cublasDestroy(dgemm::cublasHandle));

}
