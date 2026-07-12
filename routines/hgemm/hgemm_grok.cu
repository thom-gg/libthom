#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

#define BLOCK_M 128
#define BLOCK_N 128
#define BLOCK_K 64
#define WARPS_M 4
#define WARPS_N 2
#define NUM_WARPS (WARPS_M*WARPS_N)
#define THREADS (NUM_WARPS*32)
#define WARP_M 32
#define WARP_N 64
#define A_VECS (BLOCK_K/8)
#define B_VECS (BLOCK_N/8)

__device__ __forceinline__ int swz(int vec, int row) {
    return (vec & ~7) | ((vec & 7) ^ (row & 7));
}
__device__ __forceinline__ uint32_t smem_addr(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

struct AFrag { uint32_t r[2][4]; };
struct BFrag { uint32_t r[8][2]; };

__device__ __forceinline__ void loadAFrag(AFrag& f, half (*As)[BLOCK_K], int cur_unused,
                                           const half (*As_buf)[BLOCK_K], int warp_row, int lane, int ksub) {
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        int quadrant = lane / 8;
        int r_in_q   = lane % 8;
        int row = warp_row * WARP_M + i * 16 + (quadrant % 2) * 8 + r_in_q;
        int vecIdx = ksub * 2 + (quadrant / 2);
        int sc = swz(vecIdx, row);
        const half* p = &As_buf[row][sc * 8];
        uint32_t addr = smem_addr(p);
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
            : "=r"(f.r[i][0]), "=r"(f.r[i][1]), "=r"(f.r[i][2]), "=r"(f.r[i][3])
            : "r"(addr));
    }
}

__device__ __forceinline__ void loadBFrag(BFrag& f, const half (*Bs_buf)[BLOCK_N],
                                           int warp_col, int lane, int ksub) {
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        int lane16 = lane % 16;
        int kb = lane16 / 8;
        int r  = lane16 % 8;
        int krow = ksub * 16 + kb * 8 + r;
        int ncol = warp_col * WARP_N + j * 8;
        int vecIdx = ncol / 8;
        int sc = swz(vecIdx, krow);
        const half* p = &Bs_buf[krow][sc * 8];
        uint32_t addr = smem_addr(p);
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
            : "=r"(f.r[j][0]), "=r"(f.r[j][1])
            : "r"(addr));
    }
}

__global__ __launch_bounds__(THREADS)
void hgemm_kernel(const __half* __restrict__ A,
                   const __half* __restrict__ B,
                   __half* __restrict__ C,
                   int M, int N, int K,
                   float alpha, float beta)
{
    extern __shared__ __align__(16) unsigned char smem[];
    half (*As)[BLOCK_M][BLOCK_K] = reinterpret_cast<half(*)[BLOCK_M][BLOCK_K]>(smem);
    half (*Bs)[BLOCK_K][BLOCK_N] = reinterpret_cast<half(*)[BLOCK_K][BLOCK_N]>(
        smem + 2 * sizeof(half) * BLOCK_M * BLOCK_K);

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warpId = tid >> 5;
    const int warp_row = warpId / WARPS_N;
    const int warp_col = warpId % WARPS_N;

    const int block_m0 = blockIdx.y * BLOCK_M;
    const int block_n0 = blockIdx.x * BLOCK_N;
    const int numKTiles = K / BLOCK_K;

    float acc[2][8][4];
    #pragma unroll
    for (int i = 0; i < 2; i++)
      #pragma unroll
      for (int j = 0; j < 8; j++)
        #pragma unroll
        for (int r = 0; r < 4; r++) acc[i][j][r] = 0.0f;

    auto loadTileA = [&](int kt, int buf) {
        #pragma unroll
        for (int it = 0; it < 4; it++) {
            int idx = tid + it * THREADS;
            int row = idx / A_VECS;
            int vec = idx % A_VECS;
            int sc  = swz(vec, row);
            const half* gptr = A + (size_t)(block_m0 + row) * K + (kt * BLOCK_K + vec * 8);
            __pipeline_memcpy_async(&As[buf][row][sc * 8], gptr, 16);
        }
    };
    auto loadTileB = [&](int kt, int buf) {
        #pragma unroll
        for (int it = 0; it < 4; it++) {
            int idx = tid + it * THREADS;
            int row = idx / B_VECS;
            int vec = idx % B_VECS;
            int sc  = swz(vec, row);
            const half* gptr = B + (size_t)(kt * BLOCK_K + row) * N + (block_n0 + vec * 8);
            __pipeline_memcpy_async(&Bs[buf][row][sc * 8], gptr, 16);
        }
    };

    int cur = 0;
    loadTileA(0, cur);
    loadTileB(0, cur);
    __pipeline_commit();

    for (int kt = 0; kt < numKTiles; kt++) {
        __pipeline_wait_prior(0);
        __syncthreads();

        int nxt = 1 - cur;
        if (kt + 1 < numKTiles) {
            loadTileA(kt + 1, nxt);
            loadTileB(kt + 1, nxt);
            __pipeline_commit();
        }

        // ---- software-pipelined ksub loop: prefetch ksub+1 fragments while MMA'ing ksub ----
        AFrag aReg[2];
        BFrag bReg[2];
        loadAFrag(aReg[0], nullptr, 0, As[cur], warp_row, lane, 0);
        loadBFrag(bReg[0], Bs[cur], warp_col, lane, 0);

        #pragma unroll
        for (int ksub = 0; ksub < BLOCK_K / 16; ksub++) {
            int rd = ksub & 1, wr = 1 - rd;
            if (ksub + 1 < BLOCK_K / 16) {
                loadAFrag(aReg[wr], nullptr, 0, As[cur], warp_row, lane, ksub + 1);
                loadBFrag(bReg[wr], Bs[cur], warp_col, lane, ksub + 1);
            }
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                        : "+f"(acc[i][j][0]), "+f"(acc[i][j][1]),
                          "+f"(acc[i][j][2]), "+f"(acc[i][j][3])
                        : "r"(aReg[rd].r[i][0]), "r"(aReg[rd].r[i][1]),
                          "r"(aReg[rd].r[i][2]), "r"(aReg[rd].r[i][3]),
                          "r"(bReg[rd].r[j][0]), "r"(bReg[rd].r[j][1]));
                }
            }
        }
        cur = nxt;
    }

    // ---- epilogue: vectorized half2 stores ----
    int groupID = lane / 4;
    int tid_in_group = lane % 4;
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int row0 = block_m0 + warp_row * WARP_M + i * 16 + groupID;
            int row1 = row0 + 8;
            int col0 = block_n0 + warp_col * WARP_N + j * 8 + tid_in_group * 2;

            half2 v0 = __floats2half2_rn(alpha * acc[i][j][0], alpha * acc[i][j][1]);
            half2 v1 = __floats2half2_rn(alpha * acc[i][j][2], alpha * acc[i][j][3]);

            half2* p0 = reinterpret_cast<half2*>(&C[(size_t)row0 * N + col0]);
            half2* p1 = reinterpret_cast<half2*>(&C[(size_t)row1 * N + col0]);

            if (beta != 0.0f) {
                half2 c0 = *p0, c1 = *p1;
                float2 c0f = __half22float2(c0), c1f = __half22float2(c1);
                v0 = __floats2half2_rn(__low2float(v0) + beta * c0f.x, __high2float(v0) + beta * c0f.y);
                v1 = __floats2half2_rn(__low2float(v1) + beta * c1f.x, __high2float(v1) + beta * c1f.y);
            }
            *p0 = v0;
            *p1 = v1;
        }
    }
}

float hgemm_grok(const __half* A, const __half* B, __half* C,
                          size_t M, size_t N, size_t K,
                          __half alpha, __half beta)
{
    dim3 block(THREADS);
    dim3 grid((unsigned)(N / BLOCK_N), (unsigned)(M / BLOCK_M));

    float alpha_f = __half2float(alpha);
    float beta_f  = __half2float(beta);

    size_t smem_bytes = 2 * (sizeof(half) * BLOCK_M * BLOCK_K +
                              sizeof(half) * BLOCK_K * BLOCK_N);

    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(hgemm_kernel,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              (int)smem_bytes);
        attr_set = true;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    hgemm_kernel<<<grid, block, smem_bytes>>>(A, B, C, (int)M, (int)N, (int)K, alpha_f, beta_f);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        fprintf(stderr, "hgemm_kernel launch error: %s\n", cudaGetErrorString(err));

    return ms;
}