set(SELECTED_SDOT_CUDA_FILES)

add_arch_cuda_source(libroutines SELECTED_SDOT_CUDA_FILES sdot_naive_atomics all)
add_arch_cuda_source(libroutines SELECTED_SDOT_CUDA_FILES sdot_cublas all)
add_arch_cuda_source(libroutines SELECTED_SDOT_CUDA_FILES sdot_fold all)
add_arch_cuda_source(libroutines SELECTED_SDOT_CUDA_FILES sdot_fold_vectorized all)
add_arch_cuda_source(libroutines SELECTED_SDOT_CUDA_FILES sdot_2kernel all)
add_arch_cuda_source(libroutines SELECTED_SDOT_CUDA_FILES sdot_warp all)

if(NOT SELECTED_SDOT_CUDA_FILES)
    message(WARNING "No sdot CUDA implementation found for ARCH=${ARCH}")
else()
    message(STATUS "Sdot CUDA sources for arch ${ARCH}: ${SELECTED_SDOT_CUDA_FILES}")
endif()

