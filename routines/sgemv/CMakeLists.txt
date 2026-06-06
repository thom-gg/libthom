set(SELECTED_SGEMV_CUDA_FILES)

add_arch_cuda_source(libroutines SELECTED_SGEMV_CUDA_FILES sgemv_naive all)
add_arch_cuda_source(libroutines SELECTED_SGEMV_CUDA_FILES sgemv_block_row all)
add_arch_cuda_source(libroutines SELECTED_SGEMV_CUDA_FILES sgemv_col1_naive all)
add_arch_cuda_source(libroutines SELECTED_SGEMV_CUDA_FILES sgemv_col2_shmem all)
add_arch_cuda_source(libroutines SELECTED_SGEMV_CUDA_FILES sgemv_col3_vecto all)
add_arch_cuda_source(libroutines SELECTED_SGEMV_CUDA_FILES sgemv_col4_trans all)
add_arch_cuda_source(libroutines SELECTED_SGEMV_CUDA_FILES sgemv_cublas all)
add_arch_cuda_source(libroutines SELECTED_SGEMV_CUDA_FILES sgemv_vecto all)


if(NOT SELECTED_SGEMV_CUDA_FILES)
    message(WARNING "No sgemv CUDA implementation found for ARCH=${ARCH}")
else()
    message(STATUS "Sgemv CUDA sources for arch ${ARCH}: ${SELECTED_SGEMV_CUDA_FILES}")
endif()

