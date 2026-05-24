set(SELECTED_REDUCTION_CUDA_FILES)

add_arch_cuda_source(libroutines SELECTED_REDUCTION_CUDA_FILES reduction_warp 80)

if(NOT SELECTED_REDUCTION_CUDA_FILES)
    message(WARNING "No reduction CUDA implementation found for ARCH=${ARCH}")
else()
    message(STATUS "Reduction CUDA sources for arch ${ARCH}: ${SELECTED_REDUCTION_CUDA_FILES}")
endif()

