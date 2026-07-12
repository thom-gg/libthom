set(ARCH "all" CACHE STRING "CUDA compute capability to compile, for example 75, 86, 89, or all")
set(ROUTINE "all" CACHE STRING "Routine to compile, for example dgemm, hgemm, reduction, sdot, sgemv, or all")

set(LIBTHOM_ROUTINES reduction sdot sgemv dgemm hgemm)

if(NOT ROUTINE STREQUAL "all" AND NOT ROUTINE IN_LIST LIBTHOM_ROUTINES)
    message(FATAL_ERROR "Unknown ROUTINE=${ROUTINE}. Expected one of: all, ${LIBTHOM_ROUTINES}")
endif()

function(libthom_routine_enabled routine output_var)
    if(ROUTINE STREQUAL "all" OR ROUTINE STREQUAL "${routine}")
        set(${output_var} TRUE PARENT_SCOPE)
    else()
        set(${output_var} FALSE PARENT_SCOPE)
    endif()
endfunction()

include(${PROJECT_SOURCE_DIR}/cmake/LibthomCudaSources.cmake)

set(LIBTHOM_ROUTINE_SOURCES
    routines.h
    common.cpp
)

libthom_routine_enabled(reduction reduction_enabled)
if(reduction_enabled)
    list(APPEND LIBTHOM_ROUTINE_SOURCES reduction/reduction.cpp)
endif()

libthom_routine_enabled(sdot sdot_enabled)
if(sdot_enabled)
    list(APPEND LIBTHOM_ROUTINE_SOURCES sdot/sdot.cpp)
endif()

libthom_routine_enabled(sgemv sgemv_enabled)
if(sgemv_enabled)
    list(APPEND LIBTHOM_ROUTINE_SOURCES sgemv/sgemv.cpp)
endif()

libthom_routine_enabled(dgemm dgemm_enabled)
if(dgemm_enabled)
    list(APPEND LIBTHOM_ROUTINE_SOURCES dgemm/dgemm.cpp)
endif()

libthom_routine_enabled(hgemm hgemm_enabled)
if(hgemm_enabled)
    list(APPEND LIBTHOM_ROUTINE_SOURCES hgemm/hgemm.cu)
endif()

add_library(libroutines STATIC
    ${LIBTHOM_ROUTINE_SOURCES}
)

foreach(routine IN LISTS LIBTHOM_ROUTINES)
    libthom_routine_enabled(${routine} routine_enabled)
    if(routine_enabled)
        string(TOUPPER "${routine}" routine_macro_name)
        target_compile_definitions(libroutines PUBLIC LIB_HAS_ROUTINE_${routine_macro_name}=1)
    endif()
endforeach()

target_compile_options(libroutines PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<COMPILE_LANGUAGE:CUDA>:-O2>
    $<$<COMPILE_LANGUAGE:CUDA>:-extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
    $<$<COMPILE_LANGUAGE:CUDA>:--keep>
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

# Enabled routine subdirectories include their own implementations using their own CMakeLists.txts
foreach(routine IN LISTS LIBTHOM_ROUTINES)
    libthom_routine_enabled(${routine} routine_enabled)
    if(routine_enabled)
        add_subdirectory(${routine})
    endif()
endforeach()
