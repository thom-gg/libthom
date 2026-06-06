set(SELECTED_SGEMV_CUDA_FILES)

add_arch_cuda_source(libroutines SELECTED_SGEMV_CUDA_FILES sgemv_naive all)


if(NOT SELECTED_SGEMV_CUDA_FILES)
    message(WARNING "No sgemv CUDA implementation found for ARCH=${ARCH}")
else()
    message(STATUS "Sdot CUDA sources for arch ${ARCH}: ${SELECTED_SGEMV_CUDA_FILES}")
endif()

