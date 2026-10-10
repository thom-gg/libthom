set(SELECTED_DGEMM_CUDA_FILES)

add_arch_cuda_source(libroutines SELECTED_DGEMM_CUDA_FILES dgemm_blocktiling all)
add_arch_cuda_source(libroutines SELECTED_DGEMM_CUDA_FILES dgemm_cpasync 80)
add_arch_cuda_source(libroutines SELECTED_DGEMM_CUDA_FILES dgemm_cublas_tc all)
add_arch_cuda_source(libroutines SELECTED_DGEMM_CUDA_FILES dgemm_final 80)
add_arch_cuda_source(libroutines SELECTED_DGEMM_CUDA_FILES dgemm_school all)
add_arch_cuda_source(libroutines SELECTED_DGEMM_CUDA_FILES dgemm_shmem all)
add_arch_cuda_source(libroutines SELECTED_DGEMM_CUDA_FILES dgemm_tensorcores 80)
add_arch_cuda_source(libroutines SELECTED_DGEMM_CUDA_FILES dgemm_triplebuf 80)
add_arch_cuda_source(libroutines SELECTED_DGEMM_CUDA_FILES dgemm_warptiling all)


if(NOT SELECTED_DGEMM_CUDA_FILES)
    message(WARNING "No dgemm CUDA implementation found for ARCH=${ARCH}")
else()
    message(STATUS "Dgemm CUDA sources for arch ${ARCH}: ${SELECTED_DGEMM_CUDA_FILES}")
endif()

