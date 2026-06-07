#include "dgemm.h"

/* Load BMxBK and BKxBN tiles into shmem than compute partial results and sum them together */
/* Assuming BM=BK=BN */

#define BM 64
#define BN 128
#define BK 16
#define WN 32
#define WM 32
#define WNITER 4
#define WMITER 4

// mma m8n8k4 f64
#define A_FRAGMENT_VALUES 1
#define B_FRAGMENT_VALUES 1
#define C_FRAGMENT_VALUES 2


__global__ void dgemm_tensorcores_kernel(const double* A, const double* B, double* C, size_t M, size_t N, size_t K,double alpha, double beta) {
    extern __shared__ double shmem[];

    double * As = shmem;
    double * Bs = shmem + (BM * BK);
    int tid = threadIdx.x;

    // position of the warp in the blocktile
    int warpIdx = tid / 32;
    int warpRow = warpIdx / (BN/WN);
    int warpCol = warpIdx % (BN/WN);
    

    // dimension of the warp subtiles
    int WSUBN = WN / WNITER;
    int WSUBM = WM / WMITER;



    // mma fragments coords
    int laneId = tid % 32;
    int groupID = laneId >> 2;
    int threadID_in_group = laneId % 4;



    double regAFragment[A_FRAGMENT_VALUES] = {0.0};
    double regBFragment[B_FRAGMENT_VALUES] = {0.0};
    // as many C fragments as mma instructions performed by each warp (and therefore each thread)
    double regCFragment[WMITER*WNITER][C_FRAGMENT_VALUES] = {0.0};

    for (int blockOffset = 0; blockOffset<K; blockOffset += BK) {
        // ====== LOAD FROM GMEM INTO SHMEM tiles ===
        // As
        for (int loadIndex = tid*2 ; loadIndex < (BM * BK) ; loadIndex += blockDim.x*2) {
            int row = loadIndex / BK;
            int col = loadIndex % BK;
         
            *reinterpret_cast<double2*>(&As[row*BK+col]) = *reinterpret_cast<const double2*>(&A[(blockIdx.x * BM + row) * K + (col + blockOffset)]);
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
        // with tensor cores.
        for (int dotIdx = 0; dotIdx < BK; dotIdx+=4) {
            // load a values into register
            for (int warpSubTileRow = 0; warpSubTileRow<WMITER; warpSubTileRow++) {
                // load this A fragment
                {
                    // a0
                    int row = groupID;
                    int col = threadID_in_group;
                    regAFragment[0] =  As[(warpRow * WM + warpSubTileRow * WSUBM + row) * BK + (dotIdx + col) ]; 
                }
                for (int warpSubTileCol = 0; warpSubTileCol<WNITER; warpSubTileCol++) {
                    // A fragment is reused
                    // load B fragment
                    {
                        // b0
                        int row = threadID_in_group;
                        int col = groupID;
                        regBFragment[0] = Bs[(dotIdx + row) * BN + (warpCol * WN + warpSubTileCol * WSUBN + col)];

                    }
                    // have WMITER*WNITER C fragments that are reused always
                    // perform MMA
                    asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0, %1}, {%2}, {%3}, {%4, %5};"
                                : "=d"(regCFragment[warpSubTileRow*WNITER + warpSubTileCol][0]), "=d"(regCFragment[warpSubTileRow*WNITER + warpSubTileCol][1])
                                : "d"(regAFragment[0]), "d"(regBFragment[0]),
                                  "d"(regCFragment[warpSubTileRow*WNITER + warpSubTileCol][0]), "d"(regCFragment[warpSubTileRow*WNITER + warpSubTileCol][1])
                                : "memory"
                                );
                }
            }
            
        }
        // sync before writing in SHMEM again
        __syncthreads();
    }

    // ============= WRITING FINAL RESULTS BACK INTO GMEM
    for (int warpSubTileRow = 0; warpSubTileRow<WMITER; warpSubTileRow++) {
        for (int warpSubTileCol = 0; warpSubTileCol<WNITER; warpSubTileCol++) {
            {
                // c0
                int row = groupID;
                int col = (threadID_in_group * 2) + (0 & 0x1); // formula from PTX docs idk
                int globalMemRow = blockIdx.x * BM + warpRow * WM + warpSubTileRow * WSUBM + row;
                int globalMemCol = blockIdx.y * BN + warpCol * WN + warpSubTileCol * WSUBN + col;
                C[globalMemRow * N + globalMemCol] = alpha *  regCFragment[warpSubTileRow*WNITER + warpSubTileCol][0] + beta * C[globalMemRow * N + globalMemCol];

            }
            {
                // c1
                int row = groupID;
                int col = (threadID_in_group * 2) + (1 & 0x1); // formula from PTX docs idk
                int globalMemRow = blockIdx.x * BM + warpRow * WM + warpSubTileRow * WSUBM + row;
                int globalMemCol = blockIdx.y * BN + warpCol * WN + warpSubTileCol * WSUBN + col;
                C[globalMemRow * N + globalMemCol] = alpha *  regCFragment[warpSubTileRow*WNITER + warpSubTileCol][1] + beta * C[globalMemRow * N + globalMemCol];

            }
           
        }
    }

   

}

/* Not handling edge cases, only works for multiples of BLOCKSIZE */
float dgemm_tensorcores(const double* A, const double* B, double* C, size_t M, size_t N, size_t K,double alpha, double beta) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);


    int nbWarps = (BM / WM) * (BN / WN);
    int threadsPerBlock = nbWarps * 32;
    dim3 nbBlocks((M/ BM), (N / BN));

    size_t shmemSize = ((BM * BK) + (BK * BN)) * sizeof(double);
    dgemm_tensorcores_kernel<<<nbBlocks, threadsPerBlock, shmemSize>>>(A,B,C,M,N,K,alpha,beta);

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