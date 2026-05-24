cmake_minimum_required(VERSION 3.28)
project(libthom LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

add_subdirectory(routines)

add_executable(${PROJECT_NAME} 
    main.cpp
)

target_link_libraries(${PROJECT_NAME} PRIVATE libroutines CUDA::cublas)
