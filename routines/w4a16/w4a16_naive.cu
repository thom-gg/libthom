#include "w4a16.h"

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

/* First W4A16 implementation
    Both matrices are row major
    each uint32_t value corresponds to 8 consecutive rows
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

#define MAX_M_VALUE 16 // we only care about small M size for this kernel, if we had bigger M size we would just dispatch to another kernel


__global__ void w4a16_naive_kernel(const __half * A, const uint32_t * W_int4_packed, const __half * scales, __half * C, size_t M, size_t N, size_t K, int colsPerBlock, int warpsPerBlock) {
    int warpId = threadIdx.x / 32;
    int warpLane = threadIdx.x % 32;
    int n = blockIdx.x * colsPerBlock + warpLane;

    float m_accs[MAX_M_VALUE] = {};

    int valuesToTreat = K / warpsPerBlock;
    
    int startingK = warpId * valuesToTreat;
    int endingK = startingK + valuesToTreat;

    // The reuse of A will only be among a single warp. Each lane will need the same value of A at the same time
    // therefore we can just store it in registers; and the the memory transaction will be broadcasted to the whole warp

    float w[8];
    // step of 8 because each uint32_t contains 8 int4 values from W
    for (int k=startingK; k<endingK; k+=8) {
        const uint32_t packed = W_int4_packed[(k / 8) * N + n];
        dequant_int4x8(packed, __half2float(scales[(k / GROUP_SIZE) * N + n]), w);


        #pragma unroll 
        for (int m_id = 0; m_id < MAX_M_VALUE; m_id++) {
            if (m_id >= M) {continue;}

            uint4 raw = *reinterpret_cast<const uint4 *>(&A[(m_id * K) + k]);
            const __half2 * aVals = reinterpret_cast<const __half2*>(&raw);
            
            float2 aVals0 = __half22float2(aVals[0]);
            m_accs[m_id] += aVals0.x * w[0];
            m_accs[m_id] += aVals0.y * w[1];

            float2 aVals1 = __half22float2(aVals[1]);
            m_accs[m_id] += aVals1.x * w[2];
            m_accs[m_id] += aVals1.y * w[3];

            float2 aVals2 = __half22float2(aVals[2]);
            m_accs[m_id] += aVals2.x * w[4];
            m_accs[m_id] += aVals2.y * w[5];

            float2 aVals3 = __half22float2(aVals[3]);
            m_accs[m_id] += aVals3.x * w[6];
            m_accs[m_id] += aVals3.y * w[7];
        }
    }

    // Do the reduction and store in result matrix
    extern __shared__ float partials[]; // shape warpsPerBlock x MAX_M_VALUE x 32
    
    #pragma unroll
    for (int m_id = 0; m_id < MAX_M_VALUE; m_id++) {
        if (m_id < M) {partials[(warpId * MAX_M_VALUE * 32) + (m_id * 32) + warpLane] = m_accs[m_id];}
    }
    __syncthreads();

    // warp 0 sums partials of each column
    if (warpId == 0) {
        for (int m_id = 0; m_id < M; m_id++) {
            float sum = 0.0f;
            for (int w = 0; w < warpsPerBlock; w++) {sum += partials[(w * MAX_M_VALUE * 32) + (m_id * 32) + warpLane];}
            C[m_id * N + n] = __float2half(sum);
        }
    }
    
}

float w4a16_naive(const W4A16Problem& problem) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    if (problem.M > MAX_M_VALUE) {
        fprintf(stderr, "M dimension is too high for this kernel\n");
        exit(1);
    }

    // TODO: launch configuration
    // requirements: N has to be a multiple of 32 (i dont wanna handle edge cases). K. also has to be a multiple of warpsPerBlock
    const int colsPerBlock = 32;
    const int nbBlocks = problem.N / colsPerBlock;
    const int warpsPerBlock = 8; // arbitraty, the K dimension will be split in warpsPerBlock segments

    const int threadsPerBlock = warpsPerBlock * 32 /* warpsize*/;

    size_t smemSize = warpsPerBlock * MAX_M_VALUE * 32 * sizeof(float);

    cudaEventRecord(start);

    w4a16_naive_kernel<<<nbBlocks, threadsPerBlock, smemSize>>>(problem.A, problem.W_int4_packed, problem.scales, problem.C, problem.M, problem.N, problem.K, colsPerBlock, warpsPerBlock);

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
