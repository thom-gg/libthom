add_library(libroutines STATIC
    routines.h
    reduction/reduction.cpp
    reduction/reduction-warp.cu
    scan/scan.cpp
)

target_include_directories(libroutines
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}
)
