#include "dgemm.h"

/* Load BMxBK and BKxBN tiles into shmem than compute partial results and sum them together */
/* Assuming BM=BK=BN */

#define BLOCKSIZE 32
#define TM 8
#define TN 4

__global__ void dgemm_blocktiling_kernel(const double* A, const double* B, double* C, size_t M, size_t N, size_t K,double alpha, double beta) {
    extern __shared__ double shmem[];
    double * As = shmem;
    double * Bs = shmem + (BLOCKSIZE * BLOCKSIZE);

    int localRow = threadIdx.x * TM;
    int localCol = threadIdx.y * TN;

    int threadId1d = threadIdx.x * blockDim.y + threadIdx.y;
    int threadsPerBlock = blockDim.x * blockDim.y;

    int globalRow = blockIdx.x * BLOCKSIZE + localRow;
    int globalCol = blockIdx.y * BLOCKSIZE + localCol;

    double regA[TM] = {0.0};
    double regB[TN] = {0.0};
    double tmpResults[TM*TN] = {0.0};

    for (int blockOffset = 0; blockOffset<K; blockOffset += BLOCKSIZE) {
        // Load into As 
        for (int loadIndex = threadId1d ; loadIndex < (BLOCKSIZE * BLOCKSIZE) ; loadIndex += threadsPerBlock) {
            int row = loadIndex / BLOCKSIZE;
            int col = loadIndex % BLOCKSIZE;
            As[row * BLOCKSIZE + col] = A[(blockIdx.x * BLOCKSIZE + row) * K + (col + blockOffset)];
        }

        // Load into Bs
        for (int loadIndex = threadId1d ; loadIndex < (BLOCKSIZE * BLOCKSIZE) ; loadIndex += threadsPerBlock) {
            int row = loadIndex / BLOCKSIZE;
            int col = loadIndex % BLOCKSIZE;
            Bs[row * BLOCKSIZE + col] = B[(row + blockOffset) * N + (blockIdx.y * BLOCKSIZE + col)];
        }

        __syncthreads();

        for (int dotIdx = 0; dotIdx < BLOCKSIZE; dotIdx++) {
            // Load into registers
            for (int r = 0; r<TM; r++) {
                regA[r] = As[(localRow + r)* BLOCKSIZE + (dotIdx)];
            }
            for (int c = 0; c<TN; c++) {
                regB[c] = Bs[(dotIdx) * BLOCKSIZE + (localCol + c)];
            }

            // accumulate in partial results in registers
            for (int r = 0; r<TM; r++) {
                for (int c = 0; c<TN; c++) {
                    tmpResults[r * TN + c] += regA[r] * regB[c];
                }
            }
        }
        // sync before writing in SHMEM again
        __syncthreads();
    }

    // write final results in GMEM
     for (int r = 0; r<TM; r++) {
        for (int c = 0; c<TN; c++) {
            C[(globalRow + r) * N + (globalCol + c)] = alpha * tmpResults[r * TN + c] + beta*C[(globalRow + r) * N + (globalCol + c)];
        }
    }

   

}

/* Not handling edge cases, only works for multiples of BLOCKSIZE */
float dgemm_blocktiling(const double* A, const double* B, double* C, size_t M, size_t N, size_t K,double alpha, double beta) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);


    dim3 threadsPerBlock( (BLOCKSIZE / TM) , (BLOCKSIZE / TN));
    dim3 nbBlocks((M / BLOCKSIZE),(N / BLOCKSIZE));
    size_t shmemSize = ((BLOCKSIZE * BLOCKSIZE) + (BLOCKSIZE * BLOCKSIZE)) * sizeof(double);
    dgemm_blocktiling_kernel<<<nbBlocks, threadsPerBlock, shmemSize>>>(A,B,C,M,N,K,alpha,beta);

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