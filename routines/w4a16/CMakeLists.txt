set(SELECTED_W4A16_CUDA_FILES)

add_arch_cuda_source(libroutines SELECTED_W4A16_CUDA_FILES w4a16_cublas all)
add_arch_cuda_source(libroutines SELECTED_W4A16_CUDA_FILES w4a16_naive all)


if(NOT SELECTED_W4A16_CUDA_FILES)
    message(WARNING "No w4a16 CUDA implementation found for ARCH=${ARCH}")
else()
    message(STATUS "W4a16 CUDA sources for arch ${ARCH}: ${SELECTED_W4A16_CUDA_FILES}")
endif()
