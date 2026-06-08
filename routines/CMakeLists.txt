set(ARCH "all" CACHE STRING "CUDA compute capability to compile, for example 75, 86, 89, or all")

include(${PROJECT_SOURCE_DIR}/cmake/LibthomCudaSources.cmake)

add_library(libroutines STATIC
    routines.h
    common.cpp
    reduction/reduction.cpp
    sdot/sdot.cpp
    sgemv/sgemv.cpp
    dgemm/dgemm.cpp
    hgemm/hgemm.cu
)

target_compile_options(libroutines PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-extended-lambda>
)

# Setting arch property for NVCC
if(NOT ARCH STREQUAL "all")
    set_property(TARGET libroutines PROPERTY CUDA_ARCHITECTURES ${ARCH})
endif()

target_include_directories(libroutines
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}
)

# All routine subdirectories will include their own implementations using their own CMakeLists.txts
add_subdirectory(reduction)
add_subdirectory(sdot)
add_subdirectory(sgemv)
add_subdirectory(dgemm)
add_subdirectory(hgemm)