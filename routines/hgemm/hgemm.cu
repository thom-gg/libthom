#include "hgemm.h"
#include <cstddef>
#include <fstream>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>

#define BENCH_RUNS 30
#define WARMUP_RUNS 5


#define LOWER_BOUND -5.0
#define UPPER_BOUND 5.0



namespace hgemm {
    static const std::vector<std::vector<int>> SIZES = {
        {512,512,512}, 
        {1024,1024,1024}, 
        {1536, 1536, 1536}, 
        {2048, 2048, 2048}, 
        {2560, 2560, 2560}, 
        {3072, 3072, 3072},
        {3584, 3584, 3584},
        {4096, 4096, 4096},
        {8192, 8192, 8192}
    };
    
    static __half * cpuA;
    static __half * cpuB;
    static __half * cpuC;
    static __half cpuAlpha;
    static __half cpuBeta;
    static __half * resultFromGpu;
    
    static float * tmpGpuBuffer; // Used to call curandGenerate on it, and then convert from float to half 
    static __half * gpuA;
    static __half * gpuB;
    static __half * gpuC;

    
    
    cublasHandle_t cublasHandle;
    curandGenerator_t gen;
    static cudaError_t err;
    
    using HgemmImpl = float (*)(const __half* A, const __half* B, __half* C, size_t M, size_t N, size_t K,__half alpha, __half beta);
    
    
    /* Registering implementations, only if they're compiled. They are compiled if we are using the correct Compute Capability*/
    static const std::unordered_map<std::string, HgemmImpl> hgemmImpls = {
    #if LIB_HAS(HGEMM_CUBLAS)
        {"cublas", hgemm_cublas},
    #endif
    #if LIB_HAS(HGEMM_FINAL)
        {"final", hgemm_final},
    #endif
    };
 

    
    void allocate(int M, int N, int K) {
        cpuA = (__half*)malloc(M * K * sizeof(__half));
        if (cpuA == NULL) {fprintf(stderr, "Error allocating cpuA\n");}
        cpuB = (__half*)malloc(K * N * sizeof(__half));
        if (cpuB == NULL) {fprintf(stderr, "Error allocating cpuB\n");}
        cpuC = (__half*)malloc(M * N * sizeof(__half));
        if (cpuC == NULL) {fprintf(stderr, "Error allocating cpuC\n");}
        resultFromGpu = (__half*) malloc(M*N * sizeof(__half));
        if (resultFromGpu == NULL) {fprintf(stderr,"Error allocating resultFormGpu\n");}
    
        err = cudaMalloc((void **)&gpuA, M*K*sizeof(__half));
        CUDA_CHECK(err);
        err = cudaMalloc((void **)&gpuB, K*N*sizeof(__half));
        CUDA_CHECK(err);
        err = cudaMalloc((void **)&gpuC, M*N*sizeof(__half));
        CUDA_CHECK(err);

        
        // Allocating tmp buffer used for curandGenerateUniform. We want to use the same buffer for all 3 matrices
        // so allocating the maximum size (biggest dim * 2nd biggest dim)
        int dims[3] = {M, N, K};
        std::sort(dims, dims + 3, std::greater<int>());
        int needed_elements = dims[0] * dims[1];

        err = cudaMalloc((void **)&tmpGpuBuffer, needed_elements*sizeof(float));
        CUDA_CHECK(err);
    }
    
    void freeAll() {
        free(cpuA);
        free(cpuB);
        free(cpuC);
        cudaFree(gpuA);
        cudaFree(gpuB);
        cudaFree(gpuC);
        cudaFree(tmpGpuBuffer);
    }

    void __curandGenerateHalf(__half*dst, int size) {
        CURAND_CHECK(curandGenerateUniform(gen, tmpGpuBuffer, size));

        thrust::transform(
            thrust::device_ptr<float>(tmpGpuBuffer),
            thrust::device_ptr<float>(tmpGpuBuffer + size),
            thrust::device_ptr<__half>(dst),
            [] __device__ (float x) { return __float2half(x); }
        );
    }
    
    
    void resetVectors(int M, int N, int K) {
        // fill randomly using cuRAND directly on gpu
        __curandGenerateHalf(gpuA, M*K);
        __curandGenerateHalf(gpuB, K*N);
        __curandGenerateHalf(gpuC, M*N);

        cpuAlpha = __float2half(1.0f);
        cpuBeta = __float2half(1.0f);

        cudaDeviceSynchronize();

        // copy on cpu for reference computations
        err = cudaMemcpy(cpuA, gpuA, M*K* sizeof(__half), cudaMemcpyDeviceToHost);
        CUDA_CHECK(err);   
        err = cudaMemcpy(cpuB, gpuB, K * N * sizeof(__half), cudaMemcpyDeviceToHost);
        CUDA_CHECK(err); 
        err = cudaMemcpy(cpuC, gpuC, M * N * sizeof(__half), cudaMemcpyDeviceToHost);
        CUDA_CHECK(err);   
        cudaDeviceSynchronize();
    }
    
    // after this call cpuC will contain results from the reference implementation (CPU)
    void getReferenceResults(int M, int N, int K) {
        float alphaHalf = __half2float(cpuAlpha);
        float betaHalf = __half2float(cpuBeta);
        for (int row= 0; row<M; row++) {
            for (int col = 0; col<N; col++) {
                // inner dot product
                float sum = 0.0f;
                for (int k = 0; k<K; k++) {
                    // using instrinsics instead of directly operators
                    sum += __half2float(cpuA[row * K + k]) * __half2float(cpuB[k * N + col]);
                }
                float res = alphaHalf * sum + betaHalf * __half2float(cpuC[row * N + col]);
                cpuC[row * N + col] = __float2half(res);
            }
        }
    }


    
    void compareToReference(int M,int N, int K) {
        err = cudaMemcpy(resultFromGpu, gpuC, M*N*sizeof(__half), cudaMemcpyDeviceToHost);
        CUDA_CHECK(err);
        getReferenceResults(M,N,K);

        int countDifferent = 0;
        for (int row= 0; row<M; row++) {
            for (int col = 0; col<N; col++) {
                double diff = fabsf(__half2float(resultFromGpu[row*N+col]) - __half2float(cpuC[row*N+col]));
                double abs_ref = fabsf(__half2float(cpuC[row*N+col]));
                double rel_error = diff / abs_ref;
                if (rel_error > 1e-2f) {
                    countDifferent+=1;
                    printf("row %d col %d | GPU: %f | CPU: %f | rel_error: %f\n",
                        row, col, __half2float(resultFromGpu[row*N+col]), __half2float(cpuC[row*N+col]), rel_error);
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
    float runner(HgemmImpl impl, std::vector<int> sizes, int nb_runs, int nb_warmup_runs, bool checkResult, bool printEach, int argc, char * argv[]) {
        int M = sizes[0];
        int N = sizes[1];
        int K = sizes[2];
        
        float sum_gflops = 0.0f;
        allocate(M, N, K);
        resetVectors(M, N, K);

        for (int i = -1*nb_warmup_runs; i<nb_runs; i++) {
            float time_ms = impl(gpuA, gpuB, gpuC, M, N, K, cpuAlpha, cpuBeta);
            CUDA_CHECK(cudaGetLastError());
            
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




void hgemm_handler(RunMode runMode, int argc, char * argv[]) {
    CURAND_CHECK(curandCreateGenerator(&hgemm::gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(hgemm::gen, 1234ULL));
    CUBLAS_CHECK(cublasCreate(&hgemm::cublasHandle));

    switch (runMode) {
        case RunMode::TEST:
            Handlers<hgemm::HgemmImpl, 3, BENCH_RUNS, WARMUP_RUNS>::test_handler(argc, argv, hgemm::hgemmImpls, hgemm::runner);
            break;
        case RunMode::BENCH:
            Handlers<hgemm::HgemmImpl, 3, BENCH_RUNS, WARMUP_RUNS>::bench_handler(argc, argv, hgemm::hgemmImpls, hgemm::runner, hgemm::SIZES, hgemm::exportCSV);
            break;
    }

    CURAND_CHECK(curandDestroyGenerator(hgemm::gen));
    CUBLAS_CHECK(cublasDestroy(hgemm::cublasHandle));

}
