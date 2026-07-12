set(SELECTED_HGEMM_CUDA_FILES)

add_arch_cuda_source(libroutines SELECTED_HGEMM_CUDA_FILES hgemm_cublas all)
add_arch_cuda_source(libroutines SELECTED_HGEMM_CUDA_FILES hgemm_final 80)
add_arch_cuda_source(libroutines SELECTED_HGEMM_CUDA_FILES hgemm_bruce 80)
add_arch_cuda_source(libroutines SELECTED_HGEMM_CUDA_FILES hgemm_bruce2 80)
add_arch_cuda_source(libroutines SELECTED_HGEMM_CUDA_FILES hgemm_better_reuse 80)
add_arch_cuda_source(libroutines SELECTED_HGEMM_CUDA_FILES hgemm_grok 80)

if(NOT SELECTED_HGEMM_CUDA_FILES)
    message(WARNING "No hgemm CUDA implementation found for ARCH=${ARCH}")
else()
    message(STATUS "Hgemm CUDA sources for arch ${ARCH}: ${SELECTED_HGEMM_CUDA_FILES}")
endif()

