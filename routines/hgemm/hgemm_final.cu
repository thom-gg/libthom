#include "hgemm.h"

/* Load BMxBK and BKxBN tiles into shmem than compute partial results and sum them together */
/* Assuming BM=BK=BN */

#define BM 64
#define BN 128
#define BK 16
#define WN 32
#define WM 32

#define WSUBM 8 
#define WSUBN 8 

#define WNITER (WN / WSUBN)
#define WMITER (WM / WSUBM)


// mma m8n8k4 f16
#define A_FRAGMENT_VALUES 2 // 2 __half2 per thread
#define B_FRAGMENT_VALUES 2 // 2 __half2 per thread
#define C_FRAGMENT_VALUES 4 // 4 __half2 per thread

// resolvign bank conflicts
#define PADDING_AS 4
#define PADDING_BS 8


// load a tile from GMEM to SHMEM using cp.async instructions
// As and Bs should already be the correct ones of the two buffers we have in memory
// does not wait, only issue correct load instruction
__device__ void loadTile(int tid, int K, int N, __half * As, const __half * A, __half * Bs, const __half * B, int blockOffset, int ldaShared, int ldbShared) {
    if (blockOffset >= K) {  // nothing to load (this call was issued for double buffering and we are at the end of the K dimension)
        return;
    }
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

__global__ void hgemm_final_kernel(const __half* A, const __half* B, __half* C, size_t M, size_t N, size_t K,__half alpha, __half beta) {
    extern __shared__ __align__(16) __half shmem[];

    // As of size BM x BK and we have 2 of those (double buffering)
    int ldaShared = BK+PADDING_AS;
    int ldbShared = BN+PADDING_BS;
    __half * As = shmem;
    __half * Bs = shmem + 2* (BM * ldaShared);
    int tid = threadIdx.x;

    // position of the warp in the blocktile
    int warpIdx = tid / 32;
    int warpRow = warpIdx / (BN/WN);
    int warpCol = warpIdx % (BN/WN);

    __half2 alpha2 = make_half2(alpha,alpha);
    __half2 beta2 = make_half2(beta,beta);

    

    // dimension of the warp subtiles
    // int WSUBN = WN / WNITER;
    // int WSUBM = WM / WMITER;



    // mma fragments coords
    int laneId = tid % 32;
    int groupID = laneId >> 2;
    int threadID_in_group = laneId % 4;


    // storing __half2 as uint32_t  
    uint32_t regAFragment[A_FRAGMENT_VALUES];
    uint32_t regBFragment[B_FRAGMENT_VALUES];
    // as many C fragments as mma instructions performed by each warp (and therefore each thread)
    uint32_t regCFragment[WMITER*WNITER][C_FRAGMENT_VALUES] = {};

    // load  very first tile 
    {
        int selectedBuffer = 0;
        loadTile(tid, K, N, &As[selectedBuffer * (BM * ldaShared)], A, &Bs[selectedBuffer * (BK * ldbShared)], B, 0, ldaShared, ldbShared); 
        asm volatile("cp.async.wait_group 0 ;");
        __syncthreads();
    }
    
    for (int blockOffset = 0; blockOffset<K; blockOffset += BK) {
        int kTile = blockOffset/BK;
        int current_buf = kTile % 2; // 0 or 1
        int next_buf = (kTile+1) %2; // 0 or 1

        // load next buffer into shmem
        loadTile(tid, K, N, &As[next_buf * (BM* ldaShared)], A, &Bs[next_buf * (BK*ldbShared)], B, blockOffset + BK, ldaShared, ldbShared);
        // wait for current buffer (literally wait until only 1 group is pending (i.e wait until only next buffer is pending))
        



        // compute current buffer
        // ======== OK NOW WE DO SOME COMPUTE 
        // with tensor cores.
        for (int dotIdx = 0; dotIdx < BK; dotIdx+=4) {
            // load a values into register
            for (int warpSubTileRow = 0; warpSubTileRow<WMITER; warpSubTileRow++) {
                // load this A fragment
                int row = laneId < 16 ? laneId % 4 : (laneId % 4) + 4;
                {
                    // a0-a1 (packed together into a __half2)
                    int col = 0; // to read a0
                    regAFragment[0] = *reinterpret_cast<uint32_t*>(&As[(current_buf * (BM*ldaShared)) + (warpRow * WM + warpSubTileRow * WSUBM + row) * ldaShared + (dotIdx + col) ]); 
                }
                {
                    // a2-a3 (packed together into a __half2)
                    int col = 2; // to read a2
                    regAFragment[1] =  *reinterpret_cast<uint32_t*>(&As[(current_buf * (BM*ldaShared)) + (warpRow * WM + warpSubTileRow * WSUBM + row) * ldaShared + (dotIdx + col) ]); 
                }
                for (int warpSubTileCol = 0; warpSubTileCol<WNITER; warpSubTileCol++) {
                    // A fragment is reused
                    // load B fragment
                    int row = laneId % 4;
                    {
                        // b0-b1 packed together into a __half2
                        int col = laneId < 16 ? 0 : 4; // for b0
                        regBFragment[0] = *reinterpret_cast<uint32_t*>(&Bs[(current_buf * (BK*ldbShared)) + (dotIdx + row) * ldbShared + (warpCol * WN + warpSubTileCol * WSUBN + col)]);
                    }
                    {
                        // b2-b3 packed together into a __half2
                        int col = laneId < 16 ? 2 : 6; // for b2
                        regBFragment[1] = *reinterpret_cast<uint32_t*>(&Bs[(current_buf * (BK*ldbShared)) + (dotIdx + row) * ldbShared + (warpCol * WN + warpSubTileCol * WSUBN + col)]);
                    }
                    // have WMITER*WNITER C fragments that are reused always
                    // perform MMA
                    asm volatile("mma.sync.aligned.m8n8k4.row.col.f16.f16.f16.f16 {%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%8,%9,%10,%11};"
                                : "=r"(regCFragment[warpSubTileRow*WNITER + warpSubTileCol][0]), "=r"(regCFragment[warpSubTileRow*WNITER + warpSubTileCol][1]),"=r"(regCFragment[warpSubTileRow*WNITER + warpSubTileCol][2]),"=r"(regCFragment[warpSubTileRow*WNITER + warpSubTileCol][3])
                                : "r"(regAFragment[0]), "r"(regAFragment[1]), "r"(regBFragment[0]),"r"(regBFragment[1]),
                                  "r"(regCFragment[warpSubTileRow*WNITER + warpSubTileCol][0]), "r"(regCFragment[warpSubTileRow*WNITER + warpSubTileCol][1]), "r"(regCFragment[warpSubTileRow*WNITER + warpSubTileCol][2]), "r"(regCFragment[warpSubTileRow*WNITER + warpSubTileCol][3])
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
                // write the two values with a vectorized write
                // coords of c0, c1 is just c0 + 1
                int row = laneId < 16 ? laneId % 4 : (laneId %4) +4;
                int col = 0; // first __half2 is at col0, second at col2, third at col4 and last at col6
                __half2 h0 = *reinterpret_cast<__half2*>(regCFragment[warpSubTileRow*WNITER + warpSubTileCol][0]);
                __half2 h1 = *reinterpret_cast<__half2*>(regCFragment[warpSubTileRow*WNITER + warpSubTileCol][1]);
                __half2 h2 = *reinterpret_cast<__half2*>(regCFragment[warpSubTileRow*WNITER + warpSubTileCol][2]);
                __half2 h3 = *reinterpret_cast<__half2*>(regCFragment[warpSubTileRow*WNITER + warpSubTileCol][3]);

                int globalMemRow = blockIdx.x * BM + warpRow * WM + warpSubTileRow * WSUBM + row;
                int globalMemCol = blockIdx.y * BN + warpCol * WN + warpSubTileCol * WSUBN + col;

                // alpha * regC + beta * C
                h0 = __hadd2(__hmul2(alpha2, h0), __hmul2(beta2, *reinterpret_cast<__half2*>(&C[globalMemRow * N + globalMemCol+0])));
                h1 = __hadd2(__hmul2(alpha2, h1), __hmul2(beta2, *reinterpret_cast<__half2*>(&C[globalMemRow * N + globalMemCol+2])));
                h2 = __hadd2(__hmul2(alpha2, h2), __hmul2(beta2, *reinterpret_cast<__half2*>(&C[globalMemRow * N + globalMemCol+4])));
                h3 = __hadd2(__hmul2(alpha2, h3), __hmul2(beta2, *reinterpret_cast<__half2*>(&C[globalMemRow * N + globalMemCol+6])));

                uint4 out = make_uint4(
                    reinterpret_cast<uint32_t&>(h0),
                    reinterpret_cast<uint32_t&>(h1),
                    reinterpret_cast<uint32_t&>(h2),
                    reinterpret_cast<uint32_t&>(h3)
                );

                // write the 4 __half2 (i.e 8 __half => 128bits in memory)
                *reinterpret_cast<uint4*>(&C[globalMemRow * N + globalMemCol]) = out;
            }    
        }
    }

   

}

/* Not handling edge cases, only works for multiples of BLOCKSIZE */
float hgemm_final(const __half* A, const __half* B, __half* C, size_t M, size_t N, size_t K,__half alpha, __half beta) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);


    int nbWarps = (BM / WM) * (BN / WN);
    int threadsPerBlock = nbWarps * 32;
    dim3 nbBlocks((M/ BM), (N / BN));

    // x2 for double buffering
    size_t shmemSizeForA = 2 * (BM * (BK + PADDING_AS)) * sizeof(__half); // in each line we add 4 elements of padding
    size_t shmemSizeForB = 2 * (BK * (BN + PADDING_BS)) * sizeof(__half);
    size_t shmemSize = shmemSizeForA + shmemSizeForB;
    cudaError_t err = cudaFuncSetAttribute(
        hgemm_final_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shmemSize
    );
    
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaFuncSetAttribute failed: %s\n", cudaGetErrorString(err));
    }

    hgemm_final_kernel<<<nbBlocks, threadsPerBlock, shmemSize>>>(A,B,C,M,N,K,alpha,beta);

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