#pragma once

#include "../routines.h"

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))


#if LIB_HAS(DGEMM_BLOCKTILING)
float dgemm_blocktiling(const double* A, const double* B, double* C, size_t M, size_t N, size_t K,double alpha, double beta);
#endif

#if LIB_HAS(DGEMM_CPASYNC)
float dgemm_cpasync(const double* A, const double* B, double* C, size_t M, size_t N, size_t K,double alpha, double beta);
#endif

#if LIB_HAS(DGEMM_CUBLAS_TC)
float dgemm_cublas_tc(const double* A, const double* B, double* C, size_t M, size_t N, size_t K,double alpha, double beta);
#endif

#if LIB_HAS(DGEMM_FINAL)
float dgemm_final(const double* A, const double* B, double* C, size_t M, size_t N, size_t K,double alpha, double beta);
#endif

#if LIB_HAS(DGEMM_SCHOOL)
float dgemm_school(const double* A, const double* B, double* C, size_t M, size_t N, size_t K,double alpha, double beta);
#endif

#if LIB_HAS(DGEMM_SHMEM)
float dgemm_shmem(const double* A, const double* B, double* C, size_t M, size_t N, size_t K,double alpha, double beta);
#endif

#if LIB_HAS(DGEMM_TENSORCORES)
float dgemm_tensorcores(const double* A, const double* B, double* C, size_t M, size_t N, size_t K,double alpha, double beta);
#endif

#if LIB_HAS(DGEMM_TRIPLEBUF)
float dgemm_triplebuf(const double* A, const double* B, double* C, size_t M, size_t N, size_t K,double alpha, double beta);
#endif

#if LIB_HAS(DGEMM_WARPTILING)
float dgemm_warptiling(const double* A, const double* B, double* C, size_t M, size_t N, size_t K,double alpha, double beta);
#endif