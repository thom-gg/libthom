# DGEMM
Some work on the DGEMM routine (double precision general matrix matrix multiplication), initially as a school project on a Tesla K40 (sm_35), that I wanted to re-do at home afterwards on a A100 (sm_80):

## Implementations

- **dgem_cublas_tc.cu**: reference implementation
- **dgemm_school.cu**: implementation from the school project on the Tesla K40
- **dgemm_shmem.cu**: naive smem kernel: dividing C into 2D tiles, launching one thread block per tile, then splitting the K dimensions in parts of size BK that we load into SMEM to compute partial results.
- **dgemm_blocktiling.cu**: adding a level of tiling by having every thread compute a small 2d subtile of elements, to increase data reuse.
- **dgemm_warptiling.cu**: tiling around warps
- **dgemm_tensorcores.cu**: using mma instructions for Tensor Cores
- **dgemm_cpasync.cu**: double buffering kernel using cp.async instructions (hence the name)
- **dgemm_final.cu**: fixing bank conflicts by adding padding to smem layout
- **dgemm_triplebuf.cu**: experimentation with triple buffering that did not improve performance as expected

## Blog post

All the explanations and analysis: 
https://thom.gg/fighting-cublas-dgemm