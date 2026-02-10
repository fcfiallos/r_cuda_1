cmake_minimum_required(VERSION 3.16)
 
project(Cuda01)
 
enable_language(CXX CUDA)
 
find_package(fmt CONFIG REQUIRED)
find_package(SFML COMPONENTS Graphics Window System CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)
 
add_executable(cuda_fractal
    main.cpp
    kernel.cu
    palette.cpp
)
 
target_include_directories(cuda_fractal PUBLIC
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
 
target_link_libraries(cuda_fractal PRIVATE
    fmt::fmt
    SFML::Graphics SFML::Window SFML::System
    CUDA::cuda_driver
    CUDA::nvrtc
    CUDA::cudart
)