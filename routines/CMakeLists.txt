set(ARCH "all" CACHE STRING "CUDA compute capability to compile, for example 75, 86, 89, or all")

include(${PROJECT_SOURCE_DIR}/cmake/LibthomCudaSources.cmake)

add_library(libroutines STATIC
    routines.h
    reduction/reduction.cpp
    scan/scan.cpp
)

if(NOT ARCH STREQUAL "all")
    set_property(TARGET libroutines PROPERTY CUDA_ARCHITECTURES ${ARCH})
endif()

target_include_directories(libroutines
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}
)

add_subdirectory(reduction)
