#include "hgemm.h"



// mma m16n8k16 f16
#define MMA_M 16
#define MMA_N 8
#define MMA_K 16
#define A_FRAGMENT_VALUES 4 // 4 __half2 per thread
#define B_FRAGMENT_VALUES 2 // 2 __half2 per thread
#define C_FRAGMENT_VALUES 2 // 2 __half2 per thread

/* WSUBN ET WSUBM doivent couvrir la taille d'une instruction mma*/
#define WSUBM MMA_M 
#define WSUBN MMA_N



// Padding to resolve bank conflicts
#define PADDING_AS 0
#define PADDING_BS 0



// constraints:
// sizes must be multiples of 8 (for loading from gmem to smem we dont take care of edge cases if not multiple of 8)

template<int GROUP_SIZE_M>
__device__ __forceinline__ void getBlockCoords(int& blockRow, int& blockCol, int numBlocksM, int numBlocksN) {
    int pid = blockIdx.x; // la grille sera 1D maintenant

    int numPidInGroup = GROUP_SIZE_M * numBlocksN;
    int groupId = pid / numPidInGroup;
    int firstBlockRow = groupId * GROUP_SIZE_M;
    int groupSizeM = min(numBlocksM - firstBlockRow, GROUP_SIZE_M);

    // The last group can contain fewer rows than GROUP_SIZE_M.
    const int localPid = pid - groupId * numPidInGroup;
    blockRow = firstBlockRow + (localPid % groupSizeM);
    blockCol = localPid / groupSizeM;
}


// Wrapper device function that issues a "cp.async.wait_group N" instruction. it means it will wait until N or fewer groups are pending
__device__ __forceinline__ void cp_async_wait(int pending_groups) {
    if      (pending_groups == 0) asm volatile("cp.async.wait_group 0;");
    else if (pending_groups == 1) asm volatile("cp.async.wait_group 1;");
    else if (pending_groups == 2) asm volatile("cp.async.wait_group 2;");
    else if (pending_groups == 3) asm volatile("cp.async.wait_group 3;");
    else if (pending_groups == 4) asm volatile("cp.async.wait_group 4;");
    else if (pending_groups == 5) asm volatile("cp.async.wait_group 5;");
    else if (pending_groups == 6) asm volatile("cp.async.wait_group 6;");
    else if (pending_groups == 7) asm volatile("cp.async.wait_group 7;");

}

__device__ __forceinline__ int swizzleBCol(int row, int col) {
    // ne touche que les bits 3-5 de col (index de chunk de 8 halfs)
    return col ^ ((row & 7) << 3);
}

__device__ __forceinline__ int swizzleACol(int row, int col) {
    // col est un multiple de 8 (index de chunk * 8), BK/8 = 4 chunks
    int chunkIndex = col >> 3;
    int mask = (row >> 1) & 3;      // ignore parity bit because two consecutive rows use the same swizzling. '& 3' is basically modulo 4 
    int physChunk = chunkIndex ^ mask;
    return physChunk << 3;
}


// load a tile from GMEM to SMEM using cp.async instructions
// Even if doing double buffering, the As and Bs variables passed to this method should be the correct buffers for the load we want to issue. (this function is not going to determine it by itself)
// Does not wait, only issue correct load instructions
template <int BM, int BN, int BK>
__device__ __forceinline__ void loadTile(int tid, int K, int N, __half * As, const __half * A, __half * Bs, const __half * B, int blockOffset, int ldaShared, int ldbShared, int blockRow, int blockCol) {
    if (blockOffset >= K) { 
        /* Nothing to load, this call was typically issued to load the next tile while computing the last tile, i.e there's no next tile*/
        return;
    }

    /* The maximum size we can copy using the cp.async instruction is 16 bytes, which is 8 fp16 
    That's why we loop on chunks of 8 items to load them in as few instructions as possible */

    // 1 - Loading a tile of A into SMEM (As)
    for (int loadIndex = tid*8 ; loadIndex < (BM * BK) ; loadIndex += blockDim.x*8) {
        int row = loadIndex / BK;
        int col = loadIndex % BK;

        int physCol = swizzleACol(row, col); 
        size_t aAddr = __cvta_generic_to_shared(&As[row*ldaShared+physCol]);

        asm volatile ("cp.async.cg.shared.global [%0], [%1], 16;" /* 16 bytes -> 8 fp16*/
                :: "l"(aAddr), // dst 
                    "l"(&A[(blockRow* BM + row) * K + (col + blockOffset)]) // src
        );
    }

    // 2 - Loading a tile of B into SMEM (Bs)
    for (int loadIndex = tid*8 ; loadIndex < (BK * BN) ; loadIndex += blockDim.x*8) {
        int row = loadIndex / BN;
        int col = loadIndex % BN;

        int physCol = swizzleBCol(row, col);              // col: multiple de 8


        size_t bAddr = __cvta_generic_to_shared(&Bs[row*ldbShared+physCol]);

        asm volatile ("cp.async.cg.shared.global [%0], [%1], 16;" /* 16 bytes -> 8 fp16*/
                        :: "l"(bAddr), // dst 
                            "l"(&B[(row + blockOffset) * N + (blockCol* BN + col)]) // src
        );
    }

    /* Commit these issued cp.async instructions into a cp.async group */
    asm volatile("cp.async.commit_group ;");
}


template<int BM, int BN, int BK, int WM, int WN, int N_STAGES, int GROUP_SIZE_M>
__global__ void hgemm_better_reuse_kernel(const __half* A, const __half* B, __half* C, size_t M, size_t N, size_t K,__half alpha, __half beta) {
    
    constexpr int WNITER = (WN / WSUBN);
    constexpr int WMITER = (WM / WSUBM);
    
    extern __shared__ __align__(16) __half shmem[];

    int numBlocksM = M / BM;
    int numBlocksN = N / BN;
    int blockRow, blockCol;
    getBlockCoords<GROUP_SIZE_M>(blockRow, blockCol, numBlocksM, numBlocksN);

    /* Use lda and ldb variables to support the use of padding to avoid bank conflicts */
    int ldaShared = BK+PADDING_AS;
    int ldbShared = BN+PADDING_BS;

    /* We do multi-stage buffering so we need to reserve twice the size of 1 A tile for `As` */
    __half * As = shmem;
    __half * Bs = shmem + N_STAGES * (BM * ldaShared);


    int tid = threadIdx.x;

    // position of the warp in the blocktile
    int warpIdx = tid / 32;
    int warpRow = warpIdx / (BN/WN);
    int warpCol = warpIdx % (BN/WN);

    // constants
    __half2 alpha2 = make_half2(alpha,alpha);
    __half2 beta2 = make_half2(beta,beta);

    // mma fragments coords
    int laneId = tid % 32;
    int groupID = laneId >> 2;
    int threadID_in_group = laneId % 4;


    // storing __half2 as uint32_t  
    uint32_t regAFragment[WMITER][A_FRAGMENT_VALUES];
    uint32_t regBFragment[WNITER][B_FRAGMENT_VALUES];
    // as many C fragments as mma instructions performed by each warp (and therefore each thread)
    uint32_t regCFragment[WMITER*WNITER][C_FRAGMENT_VALUES] = {};

    // Loading the very first tiles before starting the double buffering loop 
    {
        /* issue first copy instructions (N_STAGES-1, because one will be issued at the beginning of the following loop)*/
        #pragma unroll
        for (int selectedBuffer = 0; selectedBuffer < N_STAGES-1 ; selectedBuffer++) {
            loadTile<BM,BN,BK>(tid, K, N, &As[selectedBuffer * (BM * ldaShared)], A, &Bs[selectedBuffer * (BK * ldbShared)], B, selectedBuffer*BK, ldaShared, ldbShared, blockRow, blockCol); 
        }
        // Wait until at least one group is finished, which means (N_STAGES-2) groups or less are still pending
        cp_async_wait(N_STAGES-2);
        __syncthreads();
    }
    
    for (int blockOffset = 0; blockOffset<K; blockOffset += BK) {
        int kTile = blockOffset/BK;
        int current_buf = kTile % N_STAGES; // [0,N_STAGES[
        // we want to load the furthest possible buffer. (i.e kTile+N_STAGES-1) 
        int buf_to_load = (kTile+N_STAGES-1) % N_STAGES; // [0,N_STAGES[

        // issue load of next buffer into shmem
        loadTile<BM,BN,BK>(tid, K, N, &As[buf_to_load * (BM* ldaShared)], A, &Bs[buf_to_load * (BK*ldbShared)], B, blockOffset + (N_STAGES-1) * BK, ldaShared, ldbShared, blockRow, blockCol);
        

        /* ========= Starting to do some compute with Tensor Cores */
        for (int dotIdx = 0; dotIdx < BK; dotIdx+=MMA_K) {


            #pragma unroll /* Load all A fragments */
            for (int warpSubTileRow = 0; warpSubTileRow<WMITER; warpSubTileRow++) {
                int rowLocal = laneId % 16;
                int colLocal= (laneId / 16) * 8;
                int fullRow = warpRow * WM + warpSubTileRow * WSUBM + rowLocal;   // ligne globale, doit matcher le store
                int logicalCol = dotIdx + colLocal;
                int physCol = swizzleACol(fullRow, logicalCol);
            
                size_t ptr64bit = __cvta_generic_to_shared(&As[(current_buf * (BM*ldaShared)) + fullRow * ldaShared + physCol]);
                uint32_t ptr = static_cast<uint32_t>(ptr64bit);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                    : "=r"(regAFragment[warpSubTileRow][0]), "=r"(regAFragment[warpSubTileRow][1]), "=r"(regAFragment[warpSubTileRow][2]), "=r"(regAFragment[warpSubTileRow][3]) 
                    : "r"(ptr)
                );
            }

            #pragma unroll /* Load all B fragments */
            for (int warpSubTileCol = 0; warpSubTileCol<WNITER; warpSubTileCol++) {
                 int row = laneId % 16; // K local : 0..15, couvre les 2 tuiles 8x8 empilées
                int logicalCol = warpCol * WN + warpSubTileCol * WSUBN; // base N (8 colonnes), pas de +col car x2 = pas de split colonne

                int physCol = swizzleBCol(dotIdx + row, logicalCol);

                size_t bAddr = __cvta_generic_to_shared(
                    &Bs[(current_buf * (BK * ldbShared)) + (dotIdx + row) * ldbShared + physCol]
                );
                uint32_t ptr = static_cast<uint32_t>(bAddr);

                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
                    : "=r"(regBFragment[warpSubTileCol][0]), "=r"(regBFragment[warpSubTileCol][1])
                    : "r"(ptr)
                );
                // {
                //     // b0-b1 packed together into a __half2
                //     int row0 = (threadID_in_group * 2) + 0; // for b0
                //     int row1 = (threadID_in_group * 2) + 1; // for b1
                //     int col =  groupID;

                //      int logicalCol = (warpCol * WN + warpSubTileCol * WSUBN + col);
                //     int physCol0 = swizzleBCol(dotIdx + row0, logicalCol);
                //     int physCol1 = swizzleBCol(dotIdx + row1, logicalCol);
                    
                //     __half2 bVals = make_half2(
                //                 Bs[(current_buf * (BK*ldbShared)) + (dotIdx + row0) * ldbShared + physCol0]
                //                 ,
                //                 Bs[(current_buf * (BK*ldbShared)) + (dotIdx + row1) * ldbShared + physCol1]);
                //     regBFragment[warpSubTileCol][0] = *reinterpret_cast<uint32_t*>(&bVals);
                // }
                // {
                //     // b2-b3 packed together
                //     int row0 = (threadID_in_group * 2) + (2 & 0x1) + 8; // for b2
                //     int row1 = (threadID_in_group * 2) + (3 & 0x1) + 8; // for b3
                //     int col =  groupID;

                //     int logicalCol = (warpCol * WN + warpSubTileCol * WSUBN + col);
                //     int physCol0 = swizzleBCol(dotIdx + row0, logicalCol);
                //     int physCol1 = swizzleBCol(dotIdx + row1, logicalCol);

                //     __half2 bVals = make_half2(
                //                 Bs[(current_buf * (BK*ldbShared)) + (dotIdx + row0) * ldbShared + physCol0]
                //                 ,
                //                 Bs[(current_buf * (BK*ldbShared)) + (dotIdx + row1) * ldbShared + physCol1]);
                //     regBFragment[warpSubTileCol][1] = *reinterpret_cast<uint32_t*>(&bVals);

                // }
            }
            #pragma unroll
            for (int warpSubTileRow = 0; warpSubTileRow<WMITER; warpSubTileRow++) {
                for (int warpSubTileCol = 0; warpSubTileCol<WNITER; warpSubTileCol++) {
                    // perform MMA
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};"
                                : "=r"(regCFragment[warpSubTileRow*WNITER + warpSubTileCol][0]), "=r"(regCFragment[warpSubTileRow*WNITER + warpSubTileCol][1])
                                : "r"(regAFragment[warpSubTileRow][0]), "r"(regAFragment[warpSubTileRow][1]), "r"(regAFragment[warpSubTileRow][2]), "r"(regAFragment[warpSubTileRow][3]), 
                                    "r"(regBFragment[warpSubTileCol][0]),"r"(regBFragment[warpSubTileCol][1]),
                                  "r"(regCFragment[warpSubTileRow*WNITER + warpSubTileCol][0]), "r"(regCFragment[warpSubTileRow*WNITER + warpSubTileCol][1])
                                );
                }
            }

            
            
        }

        // wait until at least one buffer is ready (i.e there's at maximum N_STAGES-1 in flight before jumping into next iteration
        cp_async_wait(N_STAGES-1);
        __syncthreads();

    }

    // ============= WRITING FINAL RESULTS BACK INTO GMEM
    for (int warpSubTileRow = 0; warpSubTileRow<WMITER; warpSubTileRow++) {
        for (int warpSubTileCol = 0; warpSubTileCol<WNITER; warpSubTileCol++) {
            {
                // c0 and c1
                int row = groupID;
                int col = (threadID_in_group * 2) + (0 & 0x1); // c0
                int globalMemRow = blockRow * BM + warpRow * WM + warpSubTileRow * WSUBM + row;
                int globalMemCol = blockCol* BN + warpCol * WN + warpSubTileCol * WSUBN + col;
                __half2 h0 = *reinterpret_cast<__half2*>(&regCFragment[warpSubTileRow*WNITER + warpSubTileCol][0]);
                h0 = __hadd2(__hmul2(alpha2, h0), __hmul2(beta2, *reinterpret_cast<__half2*>(&C[globalMemRow * N + globalMemCol])));
                // writing back 2 fp16 at a time
                *reinterpret_cast<uint32_t*>(&C[globalMemRow * N + globalMemCol]) = reinterpret_cast<uint32_t&>(h0);
            }
            {
                // c2 and c3
                int row = groupID + 8;
                int col = (threadID_in_group * 2) + (2 & 0x1); // c2
                int globalMemRow = blockRow * BM + warpRow * WM + warpSubTileRow * WSUBM + row;
                int globalMemCol = blockCol * BN + warpCol * WN + warpSubTileCol * WSUBN + col;
                __half2 h1 = *reinterpret_cast<__half2*>(&regCFragment[warpSubTileRow*WNITER + warpSubTileCol][1]);
                h1 = __hadd2(__hmul2(alpha2, h1), __hmul2(beta2, *reinterpret_cast<__half2*>(&C[globalMemRow * N + globalMemCol+0])));
                // writing back 2 fp16 at a time
                *reinterpret_cast<uint32_t*>(&C[globalMemRow * N + globalMemCol]) = reinterpret_cast<uint32_t&>(h1);
            }    
        }
    }

   

}

/* Not handling edge cases, only works for multiples of BLOCKSIZE */
template<int BM = 128, int BN = 128, int BK = 32, int WM = 64, int WN = 64, int N_STAGES = 2, int GROUP_SIZE_M = 4>
float hgemm_better_reuse_wrapper(const __half* A, const __half* B, __half* C, size_t M, size_t N, size_t K,__half alpha, __half beta) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);


    int nbWarps = (BM / WM) * (BN / WN);
    int threadsPerBlock = nbWarps * 32;
    dim3 nbBlocks((M/ BM) * (N / BN));


    // x N_STAGES for multi-stage buffering
    size_t shmemSizeForA = N_STAGES * (BM * (BK + PADDING_AS)) * sizeof(__half);
    size_t shmemSizeForB = N_STAGES * (BK * (BN + PADDING_BS)) * sizeof(__half);
    size_t shmemSize = shmemSizeForA + shmemSizeForB;
    cudaError_t err = cudaFuncSetAttribute(
        hgemm_better_reuse_kernel<BM,BN,BK,WM,WN,N_STAGES,GROUP_SIZE_M>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shmemSize
    );
    
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaFuncSetAttribute failed: %s\n", cudaGetErrorString(err));
    }

    hgemm_better_reuse_kernel<BM,BN,BK,WM,WN,N_STAGES,GROUP_SIZE_M><<<nbBlocks, threadsPerBlock, shmemSize>>>(A,B,C,M,N,K,alpha,beta);

    cudaEventRecord(stop);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
    }

    cudaEventSynchronize(stop);
    float ms; 
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

float hgemm_better_reuse(const __half* A, const __half* B, __half* C, size_t M, size_t N, size_t K,__half alpha, __half beta) {
    if (M == N  && N == K  && M == 512) {
        return hgemm_better_reuse_wrapper<64,64,64,32,32,4,8>(A,B,C,M,N,K,alpha,beta);
    }
    else if (M == N  && N == K  && M == 1024) {
        return hgemm_better_reuse_wrapper<64,64,64,32,32,2,8>(A,B,C,M,N,K,alpha,beta);
    }
    else if (M == N  && N == K  && M == 1536) {
        return hgemm_better_reuse_wrapper<64,128,64,32,64,2,4>(A,B,C,M,N,K,alpha,beta);
    }
    else if (M == N  && N == K  && M == 2048) {
        return hgemm_better_reuse_wrapper<128,64,32,32,64,2,4>(A,B,C,M,N,K,alpha,beta);
    }
    else if (M == N  && N == K  && M >= 2560) {
        return hgemm_better_reuse_wrapper<128,128,32,64,64,2,8>(A,B,C,M,N,K,alpha,beta);
    }


    return hgemm_better_reuse_wrapper(A,B,C,M,N,K,alpha,beta);
}

template<int... Values>
struct IntList {};

// Edit these lists to extend the search space. Every valid element of their
// Cartesian product is instantiated when this CUDA source is compiled.
using BmValues = IntList<64, 128>;
using BnValues = IntList<64,128>;
using BkValues = IntList<32, 64>;
using WmValues = IntList<32, 64>;
using WnValues = IntList<32,64>;
using StageValues = IntList<2,3,4>;
using GroupSizeValues = IntList<4, 8>;

template<int BM, int BN, int BK, int WM, int WN, int N_STAGES, int GROUP_SIZE_M>
constexpr bool is_valid_tuning_config() {
    return BM % WM == 0 && BM % MMA_M == 0 &&
           BN % WN == 0 && BN % MMA_N == 0 &&
           BK % MMA_K == 0 &&
           WM % MMA_M == 0 && WN % MMA_N == 0 &&
           ((BM / WM) * (BN / WN) * 32) <= 1024 &&
           N_STAGES >= 2 && N_STAGES <= 8 && GROUP_SIZE_M > 0;
}

template<int BM, int BN, int BK, int WM, int WN, int N_STAGES, int GROUP_SIZE_M>
void add_candidate(std::vector<HgemmBetterReuseTuningCandidate>& candidates) {
    if constexpr (is_valid_tuning_config<BM, BN, BK, WM, WN, N_STAGES, GROUP_SIZE_M>()) {
        candidates.push_back({
            "BM=" + std::to_string(BM) + " BN=" + std::to_string(BN) +
                " BK=" + std::to_string(BK) + " WM=" + std::to_string(WM) +
                " WN=" + std::to_string(WN) + " stages=" + std::to_string(N_STAGES) +
                " group_m=" + std::to_string(GROUP_SIZE_M),
            hgemm_better_reuse_wrapper<BM, BN, BK, WM, WN, N_STAGES, GROUP_SIZE_M>,
            BM, BN, BK
        });
    }
}

template<int BM, int BN, int BK, int WM, int WN, int N_STAGES, int... GROUP_SIZE_M>
void add_group_size_values(std::vector<HgemmBetterReuseTuningCandidate>& candidates, IntList<GROUP_SIZE_M...>) {
    (add_candidate<BM, BN, BK, WM, WN, N_STAGES, GROUP_SIZE_M>(candidates), ...);
}

template<int BM, int BN, int BK, int WM, int WN, int... N_STAGES>
void add_stage_values(std::vector<HgemmBetterReuseTuningCandidate>& candidates, IntList<N_STAGES...>) {
    (add_group_size_values<BM, BN, BK, WM, WN, N_STAGES>(candidates, GroupSizeValues{}), ...);
}

template<int BM, int BN, int BK, int WM, int... WN>
void add_wn_values(std::vector<HgemmBetterReuseTuningCandidate>& candidates, IntList<WN...>) {
    (add_stage_values<BM, BN, BK, WM, WN>(candidates, StageValues{}), ...);
}

template<int BM, int BN, int BK, int... WM>
void add_wm_values(std::vector<HgemmBetterReuseTuningCandidate>& candidates, IntList<WM...>) {
    (add_wn_values<BM, BN, BK, WM>(candidates, WnValues{}), ...);
}

template<int BM, int BN, int... BK>
void add_bk_values(std::vector<HgemmBetterReuseTuningCandidate>& candidates, IntList<BK...>) {
    (add_wm_values<BM, BN, BK>(candidates, WmValues{}), ...);
}

template<int BM, int... BN>
void add_bn_values(std::vector<HgemmBetterReuseTuningCandidate>& candidates, IntList<BN...>) {
    (add_bk_values<BM, BN>(candidates, BkValues{}), ...);
}

template<int... BM>
void add_bm_values(std::vector<HgemmBetterReuseTuningCandidate>& candidates, IntList<BM...>) {
    (add_bn_values<BM>(candidates, BnValues{}), ...);
}

std::vector<HgemmBetterReuseTuningCandidate> hgemm_better_reuse_tuning_candidates() {
    std::vector<HgemmBetterReuseTuningCandidate> candidates;
    add_bm_values(candidates, BmValues{});
    return candidates;
}
