#include "w4a16.h"

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

/* W4A16 implementation
    Both matrices are row major
    each uint32_t value corresponds to 8 consecutive rows

    Blocks takes BN cols of W / of C
    inside a block, each warp takes 32 consecutive cols of W / of C
    The K dimension is partitioned in slices of size BK, that we can pipeline
*/


/* Dequantize the 8 int4 values of one packed uint32 (rows k = 8p .. 8p+7 of one column).
w[i] corresponds to row 8p + i, i.e. bits 4i .. 4i+3 */
__device__ __forceinline__ void dequant_int4x8(uint32_t packed, float scale, float w[8]) {
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int q = (packed >> (4 * i)) & 0xF;
        w[i] = (float)(q - ZERO_POINT) * scale;
    }
}

/* Tile size considerations:
- BN has to be a multiple of 32 so it can easily be divided in warps. 
- N has to be a multiple of BN
- BK has to be a multiple of 8 (for the packed int4 values)
- M has to be a multiple of BK
*/
#define BN 128
#define BK 128
#define PIPELINE_STAGES 3


#define MAX_M_VALUE 16 // we only care about small M size for this kernel, if we had bigger M size we would just dispatch to another kernel


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

/* Assuming BK >= GROUP_SIZE and BK is a multiple of GROUP_SIZE for now */
__device__ __forceinline__ void load_tile(int startingK, int blockStartingCol, const __half * A, __half * A_smem, 
                                          const uint32_t * W_int4_packed, uint32_t * W_smem, const __half * scales, __half * scales_smem, int M, int N, int K) {
            
    if (startingK < K) {
      /* cp from A */
      { /* 16 bytes copy = 8 fp16*/
        for (int loadIndex = threadIdx.x * 8 ; loadIndex < MAX_M_VALUE * BK ; loadIndex += blockDim.x * 8) {
          int localRow = loadIndex / BK;
          int localCol = loadIndex  % BK;
          int globalCol = startingK + localCol;
  
          const bool validRow = localRow < M;
          const __half * src = validRow ? &A[localRow * K + globalCol] : A; /* if not validRow; A provides a valid address but we'll never read it*/
  
          size_t dst_shared_addr = __cvta_generic_to_shared(&A_smem[localRow * BK + localCol]);
          
          const int srcSize = validRow ? 16 : 0;
          asm volatile ("cp.async.ca.shared.global [%0], [%1], 16, %2;"
                :: "l"(dst_shared_addr), "l"(src), "r"(srcSize));
  
        }
      }
      /* cp from W */
      {
        for (int loadIndex = threadIdx.x * 4; loadIndex < (BN * (BK / INT4_PER_UINT32)); loadIndex += blockDim.x * 4) {
          int localRow = loadIndex / BN;
          int localCol = loadIndex % BN;
          int globalRow = (startingK / INT4_PER_UINT32) + localRow;
          int globalCol = blockStartingCol + localCol;

          size_t dst_shared_addr = __cvta_generic_to_shared(&W_smem[localRow * BN + localCol]);
          const uint32_t * src = &W_int4_packed[globalRow * N + globalCol];
          asm volatile ("cp.async.cg.shared.global [%0], [%1], 16;"
                :: "l"(dst_shared_addr), "l"(src));
        } 
      }
      /* cp from scales */
      {
        for (int loadIndex = threadIdx.x * 8; loadIndex < (BN * (BK / GROUP_SIZE)); loadIndex += blockDim.x * 8) {
          int localRow = loadIndex / BN;
          int localCol = loadIndex % BN;
          int globalRow = (startingK / GROUP_SIZE) + localRow;
          int globalCol = blockStartingCol + localCol;
          size_t dst_shared_addr = __cvta_generic_to_shared(&scales_smem[localRow * BN + localCol]);
          const __half * src = &scales[globalRow * N + globalCol];
          asm volatile ("cp.async.cg.shared.global [%0], [%1], 16;"
                :: "l"(dst_shared_addr), "l"(src));
        }
      }
    }

    asm volatile ("cp.async.commit_group ;");

}

__global__ void w4a16_pipelining_kernel(const __half * A, const uint32_t * W_int4_packed, const __half * scales, __half * C, size_t M, size_t N, size_t K) {
  int warpId = threadIdx.x / 32;
  int warpLane = threadIdx.x % 32;
  int blockStartingCol = blockIdx.x * BN;
  int colNumber = blockStartingCol + warpId * 32 + warpLane;

  __shared__ __align__(16) __half A_smem[PIPELINE_STAGES][MAX_M_VALUE * BK]; // TODO: add padding for bank conflicts?
  __shared__ __align__(16) uint32_t W_smem[PIPELINE_STAGES][BN * (BK / 8)];
  __shared__ __align__(16) __half scales_smem[PIPELINE_STAGES][BN * (BK / GROUP_SIZE)];

  /* Prologue: start the PIPELINE_STAGES-1 first copies */
  for (int currentK = 0; (currentK / BK) < PIPELINE_STAGES-1; currentK += BK) {
    int tileNumber = currentK / BK;
    load_tile(currentK, blockStartingCol, A, A_smem[tileNumber], W_int4_packed, W_smem[tileNumber], scales, scales_smem[tileNumber], M, N, K);
  }
  for (int currentK = 0 ; currentK < K ; currentK += BK) {
    /* Wait until at least  2 tiles are available: the one we need to compute on, and another one to launc a copy on */
    cp_async_wait(PIPELINE_STAGES-2);
    __syncthreads();

    /* Launch copy of the next tile */
    int tileNumber = currentK / BK;
    int tileNumberToCopy = tileNumber + PIPELINE_STAGES - 1;
    int stageIndex = tileNumberToCopy % PIPELINE_STAGES;
    load_tile(tileNumberToCopy * BK, blockStartingCol, A, A_smem[stageIndex], W_int4_packed, W_smem[stageIndex], scales, scales_smem[stageIndex], M, N, K);

    /* Work on that available tile (compute with TC)*/

  }
    
}

float w4a16_pipelining(const W4A16Problem& problem) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    if (problem.M > MAX_M_VALUE) {
        fprintf(stderr, "M dimension is too high for this kernel\n");
        exit(1);
    }

    // TODO: launch configuration
    const int threadsPerBlock = BN; /* As many threads as cols of W a block is taking care of*/
    const int nbBlocks = problem.N / BN; /* N has to be a multiple of BN */

    size_t smemSize = warpsPerBlock * MAX_M_VALUE * 32 * sizeof(float);

    cudaEventRecord(start);

    w4a16_pipelining_kernel<<<nbBlocks, threadsPerBlock, smemSize>>>(problem.A, problem.W_int4_packed, problem.scales, problem.C, problem.M, problem.N, problem.K, colsPerBlock, warpsPerBlock);

    cudaEventRecord(stop);

    cudaError_t err = cudaGetLastError();
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
