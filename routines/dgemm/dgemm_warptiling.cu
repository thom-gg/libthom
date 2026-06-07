#include "dgemm.h"

/* Load BMxBK and BKxBN tiles into shmem than compute partial results and sum them together */
/* Assuming BM=BK=BN */

#define BM 64
#define BN 128
#define BK 16
#define WN 64
#define WM 32
#define TM 4
#define TN 4
#define WNITER 1
#define WMITER ((WN * WM) / (32 * TM * TN * WNITER))


__global__ void dgemm_warptiling_kernel(const double* A, const double* B, double* C, size_t M, size_t N, size_t K,double alpha, double beta) {
    extern __shared__ double shmem[];
    // As is COL MAJOR and of dimension BMxBK
    double * As = shmem;
    double * Bs = shmem + (BM * BK);
    int tid = threadIdx.x;

    // position of the warp in the blocktile
    int warpIdx = tid / 32;
    int warpRow = warpIdx / (BN/WN);
    int warpCol = warpIdx % (BN/WN);
    
    // int WMITER = (WN * WM) / (32 * TM * TN * WNITER);


    // dimension of the warp subtiles
    int WSUBN = WN / WNITER;
    int WSUBM = WM / WMITER;

    // position of the thread inside the warp
    int threadIdxInWarp = threadIdx.x % 32;
    int threadRowInWarp = threadIdxInWarp / (WSUBN / TN);
    int threadColInWarp = threadIdxInWarp % (WSUBN / TN);

   
    double regA[TM * WMITER] = {0.0};
    double regB[TN * WNITER] = {0.0};
    double tmpResults[TM*WMITER*TN*WNITER] = {0.0};

    for (int blockOffset = 0; blockOffset<K; blockOffset += BK) {
        // ====== LOAD FROM GMEM INTO SHMEM tiles ===
        // As
        for (int loadIndex = tid*2 ; loadIndex < (BM * BK) ; loadIndex += blockDim.x*2) {
            int row = loadIndex / BK;
            int col = loadIndex % BK;
            double2 aVals = *reinterpret_cast<const double2*>(&A[(blockIdx.x * BM + row) * K + (col + blockOffset)]);
            As[row  + (col + 0) * BM] = aVals.x; 
            As[row + (col + 1) * BM] = aVals.y; 
            // As[row * BK + col] = A[(blockIdx.x * BM + row) * K + (col + blockOffset)];
        }

        // Bs
        for (int loadIndex = tid*2 ; loadIndex < (BK * BN) ; loadIndex += blockDim.x*2) {
            int row = loadIndex / BN;
            int col = loadIndex % BN;
            *reinterpret_cast<double2*>(&Bs[row*BN+col]) = *reinterpret_cast<const double2*>(&B[(row + blockOffset) * N + (blockIdx.y * BN + col)]);
            // Bs[row * BN + col] = B[(row + blockOffset) * N + (blockIdx.y * BN + col)];
        }

        __syncthreads();


        // ======== OK NOW WE DO SOME COMPUTE
        for (int dotIdx = 0; dotIdx < BK; dotIdx++) {
            // load a values into register
            for (int warpSubTileRow = 0; warpSubTileRow<WMITER; warpSubTileRow++) {
                for (int tRow = 0; tRow < TM; tRow ++) {
                    regA[warpSubTileRow * TM + tRow] = As[(warpRow * WM + threadRowInWarp*TM + warpSubTileRow * WSUBM + tRow) + (dotIdx) * BM];
                    // regA[warpSubTileRow * TM + tRow] = As[(warpRow * WM + threadRowInWarp*TM + warpSubTileRow * WSUBM + tRow) * BK + (dotIdx)];
                }
            }

            // load b values into register
            for (int warpSubTileCol = 0; warpSubTileCol<WNITER; warpSubTileCol++) {
                for (int tCol = 0; tCol< TN; tCol++) {
                    regB[warpSubTileCol * TN + tCol] = Bs[(dotIdx) * BN + (warpCol * WN + threadColInWarp*TN + warpSubTileCol * WSUBN + tCol)];
                }
            }


            // compute into local registers
            for (int warpSubTileRow = 0; warpSubTileRow<WMITER; warpSubTileRow++) {
                for (int warpSubTileCol = 0; warpSubTileCol<WNITER; warpSubTileCol++) {
                    for (int tRow = 0; tRow < TM; tRow ++) {
                        for (int tCol = 0; tCol< TN; tCol++) {
                            tmpResults[(warpSubTileRow * TM + tRow) * (TN*WNITER) + (warpSubTileCol * TN + tCol)] +=
                                    regA[warpSubTileRow * TM + tRow] * regB[warpSubTileCol * TN + tCol];
                        }
                    }
                }
            }


            
        }
        // sync before writing in SHMEM again
        __syncthreads();
    }

    // ============= WRITING FINAL RESULTS BACK INTO GMEM
    for (int warpSubTileRow = 0; warpSubTileRow<WMITER; warpSubTileRow++) {
        for (int warpSubTileCol = 0; warpSubTileCol<WNITER; warpSubTileCol++) {
            for (int tRow = 0; tRow < TM; tRow ++) {
                for (int tCol = 0; tCol< TN; tCol++) {
                    int globalMemRow = blockIdx.x * BM + warpRow * WM + warpSubTileRow * WSUBM + threadRowInWarp * TM + tRow;
                    int globalMemCol = blockIdx.y * BN + warpCol * WN + warpSubTileCol * WSUBN + threadColInWarp * TN + tCol;
                    C[globalMemRow * N + globalMemCol] = alpha * tmpResults[(warpSubTileRow * TM + tRow) * (TN*WNITER) + (warpSubTileCol * TN + tCol)] 
                                                        + beta * C[globalMemRow * N + globalMemCol];
                }
            }
        }
    }

   

}

/* Not handling edge cases, only works for multiples of BLOCKSIZE */
float dgemm_warptiling(const double* A, const double* B, double* C, size_t M, size_t N, size_t K,double alpha, double beta) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);


    int nbWarps = (BM / WM) * (BN / WN);
    int threadsPerBlock = nbWarps * 32;
    dim3 nbBlocks((M/ BM), (N / BN));

    size_t shmemSize = ((BM * BK) + (BK * BN)) * sizeof(double);
    dgemm_warptiling_kernel<<<nbBlocks, threadsPerBlock, shmemSize>>>(A,B,C,M,N,K,alpha,beta);

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