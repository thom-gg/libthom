#include "sgemv.h"


/* NOT THE OPTIMIZED ONE
This implementation is just kept to see the evolution of different implementations

For clean, commented, and optimized ones see:

------- sgemv_vecto.cu (row major) --------
------- sgemv_col3_vecto.cu (col major) --------

*/

// 1 thread per row of A (coalesced accesseses to both A and x)
__global__ void col4_trans_kernel(int M, int N, const float * alpha, const float * A, int lda, const float * x, int incx, const float * beta, float * y, int incy) {
    int tid = threadIdx.x;
    int globalIdx = tid + blockIdx.x * blockDim.x;
    int rowId = (globalIdx / 32) * 4;
    float alphaVal = *alpha;
    float betaVal = *beta;
    int warpLane = tid % 32; 

    float dot0 = 0.0f;
    float dot1 = 0.0f;
    float dot2 = 0.0f;
    float dot3 = 0.0f;
    if (rowId < M-3) {
         for (int i = warpLane; i<N; i+= 32) {
            float xVal = x[i * incx];
            float4 aValues = reinterpret_cast<const float4*>(&A[rowId + lda * i])[0];
            dot0 += aValues.x *  xVal;
            dot1 += aValues.y *  xVal;
            dot2 += aValues.z *  xVal;
            dot3 += aValues.w *  xVal;
        }
    }
    else if (rowId < M){
        for (int i = 0; i<N; i++) {
            float xVal = x[i * incx];
            dot0 += A[rowId + lda * i] *  xVal;
            if (rowId + 1 < M) {
                dot1 += A[rowId + 1 + (lda * i)] *  xVal;
            }
            if (rowId + 2 < M) {
                dot2 += A[rowId + 2 + (lda * i)] *  xVal;
            }
        }
       
    }

    // intra warp reduction
        for (int offset = 16; offset > 0; offset /= 2) {
            dot0 += __shfl_down_sync(0xffffffff, dot0, offset);
            dot1 += __shfl_down_sync(0xffffffff, dot1, offset);
            dot2 += __shfl_down_sync(0xffffffff, dot2, offset);
            dot3 += __shfl_down_sync(0xffffffff, dot3, offset);
        }

    if (warpLane == 0) {
        if (rowId < M) {
            y[rowId] = alphaVal * dot0 + betaVal * y[rowId];
        }
        if (rowId + 1 < M) {
            y[rowId + 1] = alphaVal * dot1 + betaVal * y[rowId + 1];
        }
        if (rowId + 2 < M) {
            y[rowId + 2] = alphaVal * dot2 + betaVal * y[rowId + 2];
        }
        if (rowId + 3 < M) {
            y[rowId + 3] = alphaVal * dot3 + betaVal * y[rowId + 3];
        }       
    }
    
}


float sgemv_col4_trans(int m, int n, const float * alpha, const float * A, int lda, const float * x, int incx, const float * beta, float * y, int incy, bool rowMajor) {
    if (rowMajor) {
        fprintf(stderr, "Called a column major kernel on a ROW major execution; Stopping\n");
        exit(1);
    }
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    
    float * transposedA;
    float alphaTrans = 1.0f;
    float betaTrans = 0.0f;
    cudaMalloc(&transposedA, sizeof(float) * m * n);
    cublasSgeam(
        sgemv::cublasHandle,
        CUBLAS_OP_T,   // op(A) = A transposée
        CUBLAS_OP_N,   // op(B) ignoré (beta = 0)
        n, m,          // dimensions de C (n lignes, m colonnes)
        &alphaTrans,
        A, lda,        // A et son leading dimension
        &betaTrans,
        A, m,        // B (ignoré)
        transposedA, n         // C et son leading dimension
    );

    sgemv_vecto(m, n, alpha, transposedA, n, x, incx, beta, y, incy, true);
  
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
    }
    
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float ms; 
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}