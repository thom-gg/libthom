#include "dgemm.h"

/* Load BMxBK and BKxBN tiles into smem than compute partial results and sum them together */

#define BM 64
#define BN 128
#define BK 16
#define WN 32
#define WM 32

#define WSUBM 8 
#define WSUBN 8 

#define WNITER (WN / WSUBN)
#define WMITER (WM / WSUBM)


// mma m8n8k4 f64
#define A_FRAGMENT_VALUES 1
#define B_FRAGMENT_VALUES 1
#define C_FRAGMENT_VALUES 2

// Using padding to remove bank conflicts
#define PADDING_AS 4
#define PADDING_BS 8


/* Load a tile from GMEM to SMEM using cp.async instructions
If we are doing multi-stage pipelining and have multiple buffers for As and Bs, the 
As and Bs parameters passed to this function should already be the correct buffers.
This function does not wait, only issues correct load instruction */
__device__ void loadTile(int tid, int K, int N, double * As, const double * A, double * Bs, const double * B, int blockOffset, int ldaShared, int ldbShared) {
    if (blockOffset >= K) {  // nothing to load (this call was issued for double buffering and we are at the end of the K dimension)
        return;
    }
    // As
    for (int loadIndex = tid*2 ; loadIndex < (BM * BK) ; loadIndex += blockDim.x*2) {
        int row = loadIndex / BK;
        int col = loadIndex % BK;

        size_t aAddr = __cvta_generic_to_shared(&As[row*ldaShared+col]);

        asm volatile ("cp.async.ca.shared.global [%0], [%1], 16;"
                :: "l"(aAddr), // dst 
                    "l"(&A[(blockIdx.x * BM + row) * K + (col + blockOffset)]) // src
        );
    }

    // Bs
    for (int loadIndex = tid*2 ; loadIndex < (BK * BN) ; loadIndex += blockDim.x*2) {
        int row = loadIndex / BN;
        int col = loadIndex % BN;

        size_t bAddr = __cvta_generic_to_shared(&Bs[row*ldbShared+col]);

        asm volatile ("cp.async.ca.shared.global [%0], [%1], 16;"
                        :: "l"(bAddr), // dst 
                            "l"(&B[(row + blockOffset) * N + (blockIdx.y * BN + col)]) // src
        );
    }

    asm volatile("cp.async.commit_group ;");
}

__global__ void dgemm_final_kernel(const double* A, const double* B, double* C, size_t M, size_t N, size_t K,double alpha, double beta) {
    extern __shared__ __align__(16) double shmem[];

    // As of size BM x BK and we have 2 of those (double buffering)
    int ldaShared = BK+PADDING_AS;
    int ldbShared = BN+PADDING_BS;
    double * As = shmem;
    double * Bs = shmem + 2* (BM * ldaShared);
    int tid = threadIdx.x;

    // position of the warp in the blocktile
    int warpIdx = tid / 32;
    int warpRow = warpIdx / (BN/WN);
    int warpCol = warpIdx % (BN/WN);
    


    // Values used to compute mma fragments coords
    int laneId = tid % 32;
    int groupID = laneId >> 2;
    int threadID_in_group = laneId % 4;



    double regAFragment[A_FRAGMENT_VALUES] = {0.0};
    double regBFragment[B_FRAGMENT_VALUES] = {0.0};
    // as many C fragments as mma instructions performed by each warp (and therefore each thread)
    double regCFragment[WMITER*WNITER][C_FRAGMENT_VALUES] = {0.0};

    //  Before entering the loop, load the very first tile
    {
        int selectedBuffer = 0;
        loadTile(tid, K, N, &As[selectedBuffer * (BM * ldaShared)], A, &Bs[selectedBuffer * (BK * ldbShared)], B, 0, ldaShared, ldbShared); 
        asm volatile("cp.async.wait_group 0 ;");
        __syncthreads();
    }

    // ============= COMPUTE LOOP
    for (int blockOffset = 0; blockOffset<K; blockOffset += BK) {
        int kTile = blockOffset/BK;
        int current_buf = kTile % 2; // 0 or 1
        int next_buf = (kTile+1) %2; // 0 or 1

        // Start the asynchronous load of the next buffer into SMEM
        loadTile(tid, K, N, &As[next_buf * (BM* ldaShared)], A, &Bs[next_buf * (BK*ldbShared)], B, blockOffset + BK, ldaShared, ldbShared);
        

        // compute on current buffer using tensor cores
        for (int dotIdx = 0; dotIdx < BK; dotIdx+=4) {
            // load A values into register
            for (int warpSubTileRow = 0; warpSubTileRow<WMITER; warpSubTileRow++) {
                // load this A fragment
                {
                    // a0
                    int row = groupID;
                    int col = threadID_in_group;
                    regAFragment[0] =  As[(current_buf * (BM*ldaShared)) + (warpRow * WM + warpSubTileRow * WSUBM + row) * ldaShared + (dotIdx + col) ]; 
                }
                for (int warpSubTileCol = 0; warpSubTileCol<WNITER; warpSubTileCol++) {
                    // A fragment is reused
                    // load B fragment
                    {
                        // b0
                        int row = threadID_in_group;
                        int col = groupID;
                        regBFragment[0] = Bs[(current_buf * (BK*ldbShared)) + (dotIdx + row) * ldbShared + (warpCol * WN + warpSubTileCol * WSUBN + col)];

                    }
                    // We have WMITER*WNITER C fragments that are reused always
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

        // wait for next buffer to be ready before jumping into next iteration
        asm volatile("cp.async.wait_group 0 ;");
        __syncthreads();

    }

    // ============= WRITING FINAL RESULTS BACK INTO GMEM
    for (int warpSubTileRow = 0; warpSubTileRow<WMITER; warpSubTileRow++) {
        for (int warpSubTileCol = 0; warpSubTileCol<WNITER; warpSubTileCol++) {
            {
                // write the two values with a single vectorized write
                // coords of c0, c1 is just c0 + 1
                int row = groupID;
                int col = (threadID_in_group * 2) + (0); // formula from PTX docs idk
                int globalMemRow = blockIdx.x * BM + warpRow * WM + warpSubTileRow * WSUBM + row;
                int globalMemCol = blockIdx.y * BN + warpCol * WN + warpSubTileCol * WSUBN + col;
                double2 c0c1vals = make_double2(
                    alpha *  regCFragment[warpSubTileRow*WNITER + warpSubTileCol][0] + beta * C[globalMemRow * N + globalMemCol], // c0
                    alpha *  regCFragment[warpSubTileRow*WNITER + warpSubTileCol][1] + beta * C[globalMemRow * N + globalMemCol+1] //c1
                );
                *reinterpret_cast<double2 *>(&C[globalMemRow * N + globalMemCol]) = c0c1vals;
            }      
        }
    }

   

}

float dgemm_final(const double* A, const double* B, double* C, size_t M, size_t N, size_t K,double alpha, double beta) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);


    int nbWarps = (BM / WM) * (BN / WN);
    int threadsPerBlock = nbWarps * 32;
    dim3 nbBlocks((M/ BM), (N / BN));

    // x2 for double buffering
    size_t shmemSizeForA = 2 * (BM * (BK + PADDING_AS)) * sizeof(double); // in each line we add 4 elements of padding
    size_t shmemSizeForB = 2 * (BK * (BN + PADDING_BS)) * sizeof(double);
    size_t shmemSize = shmemSizeForA + shmemSizeForB;
    cudaError_t err = cudaFuncSetAttribute(
        dgemm_final_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shmemSize
    );
    
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaFuncSetAttribute failed: %s\n", cudaGetErrorString(err));
    }

    dgemm_final_kernel<<<nbBlocks, threadsPerBlock, shmemSize>>>(A,B,C,M,N,K,alpha,beta);

    cudaEventRecord(stop);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
    }

    cudaEventSynchronize(stop);
    float ms; 
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}