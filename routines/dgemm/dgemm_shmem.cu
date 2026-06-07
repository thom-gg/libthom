#include "dgemm.h"

/* Load BMxBK and BKxBN tiles into shmem than compute partial results and sum them together */
/* Assuming BM=BK=BN */
__global__ void dgemm_shmem_kernel(const double* A, const double* B, double* C, size_t M, size_t N, size_t K,double alpha, double beta, int BLOCKSIZE) {

    extern __shared__ double shmem[];
    double * As = shmem;
    double * Bs = shmem + (BLOCKSIZE * BLOCKSIZE); // start + sizeof(As)

    int localRow = threadIdx.x / BLOCKSIZE;
    int localCol = threadIdx.x % BLOCKSIZE;

    int globalRow = blockIdx.x * BLOCKSIZE + localRow;
    int globalCol = blockIdx.y * BLOCKSIZE + localCol;

    double registerSum = 0.0;

    for (int bkStart = 0; bkStart<K; bkStart += BLOCKSIZE) {
        // load A into As (shmem)
        As[localRow * BLOCKSIZE + localCol] = A[globalRow * K + (bkStart + localCol)];

        // load B into Bs (shmem)
        Bs[localRow * BLOCKSIZE + localCol] = B[(bkStart + localRow) * N + globalCol];
        __syncthreads();

        // each thread computes its element in the tile
        for (int dotIdx = 0; dotIdx < BLOCKSIZE; dotIdx++) {
            registerSum += As[localRow * BLOCKSIZE + dotIdx] * Bs[dotIdx * BLOCKSIZE + localCol];
        }
        __syncthreads();
    }

    C[globalRow * N + globalCol] = registerSum;
}

/* Not handling edge cases, only works for multiples of BLOCKSIZE */
float dgemm_shmem(const double* A, const double* B, double* C, size_t M, size_t N, size_t K,double alpha, double beta) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);


    int BLOCKSIZE = 16;

    int threadsPerBlock = BLOCKSIZE * BLOCKSIZE;
    dim3 nbBlocks(CEIL_DIV(M,BLOCKSIZE),CEIL_DIV(N, BLOCKSIZE));

    size_t shmemSize = (sizeof(double) * BLOCKSIZE * BLOCKSIZE) + (sizeof(double) * BLOCKSIZE * BLOCKSIZE);
    dgemm_shmem_kernel<<<nbBlocks, threadsPerBlock, shmemSize>>>(A,B,C,M,N,K,alpha,beta,BLOCKSIZE);

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