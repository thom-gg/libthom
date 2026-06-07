#include "dgemm.h"

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 16;
constexpr int TM = 4;
constexpr int TN = 4;


__global__ void dgemm_cuda_kernel_2dblocktile( int M,  int N,  int K,  double alpha,
                                 const double *  A, const double *  B,
                                 double beta, double *C
                                  ) 
{
    __shared__ double localABlock[BM * BK];
    __shared__ double localBBlock[BK * BN];

    
    int num_threads_in_block = blockDim.x * blockDim.y;
    int threadLocalId = threadIdx.x;
    
    double threadResults[TM * TN] = {0.0};

    int threadRow = threadLocalId / (BN / TN);
    int threadCol = threadLocalId % (BN / TN);
    int tileStartRow = threadRow* TM;
    int tileStartCol = threadCol * TN;
    
    double registersA[TM] = {0.0};
    double registersB[TN] = {0.0};
    
    int nbBlocksKDim = (K+BK-1) / BK;

    for (int block = 0; block < nbBlocksKDim; block += 1) {
        // chargement collaboratif de la tuile de A dans la mémoire partagée
        for (int loadOffset = 0; loadOffset < BM*BK; loadOffset += num_threads_in_block) {
            int indexToLoad = loadOffset + threadLocalId;
            // indexToLoad est entre 0 et BM*BK
            int localCol = indexToLoad % BK;
            int localRow = indexToLoad / BK;

            int globalRow = localRow + (blockIdx.y * BM);
            int globalCol = ((block * BK) + localCol);
            if (globalRow < M && globalCol < K) {
                localABlock[localRow * BK + localCol] = A[globalCol + globalRow * K];
            }
            else {localABlock[localRow * BK + localCol]=0.0;}
        }

        for (int loadOffset = 0; loadOffset < BK*BN; loadOffset += num_threads_in_block) {
            int indexToLoad = loadOffset + threadLocalId;
            int localCol = indexToLoad % BN;
            int localRow = indexToLoad / BN;

            int globalCol = localCol + blockIdx.x * BN;
            int globalRow = (localRow + (block * BK));
            if (globalRow < K && globalCol < N) {
                localBBlock[localRow * BN + localCol] = B[globalCol + globalRow * N];
            }
            else {localBBlock[localRow * BN + localCol]=0.0;}
        }
        
       
        __syncthreads();

        // now do the computation


       for (int k = 0 ; k<BK; k++) {
            // load into registers
            for (int m = 0 ; m < TM; m++) {
                registersA[m] = localABlock[(m + tileStartRow) * BK + k];
            }
            for (int n = 0; n < TN; n++) {
                registersB[n] = localBBlock[(n + tileStartCol)+ k * BN];
            }

            // calculer cette tile TMxTN
            for (int m = 0; m<TM; m++) {
                for (int n = 0; n<TN; n++) {
                    threadResults[m * TN + n] += registersA[m] * registersB[n];
                }
            }
       }
       __syncthreads();
    }

    // écriture des résultats
    for (int m = 0; m<TM; m++) {
        for (int n = 0; n<TN; n++) {
            int globalRow = blockIdx.y * BM + tileStartRow + m;
            int globalCol = blockIdx.x * BN + tileStartCol + n;
            if (globalRow < M && globalCol < N) {
                C[globalCol + globalRow * N] =  beta * C[globalCol + globalRow * N] + alpha * threadResults[m * TN + n];
            }
        }
    }
    
}

/* Not handling edge cases, only works for multiples of BLOCKSIZE */
float dgemm_school(const double* A, const double* B, double* C, size_t M, size_t N, size_t K,double alpha, double beta) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);


    dim3 threadsPerBlock((BN * BM) / (TN * TM));
    dim3 nbOfBlocks((N+BN-1)/BN, (M+BM-1)/BM);

    dgemm_cuda_kernel_2dblocktile<<<nbOfBlocks,threadsPerBlock>>>(M,N,K, alpha, A, B, beta,C);

    cudaEventRecord(stop);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
    }

    cudaEventSynchronize(stop);
    float ms; 
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}