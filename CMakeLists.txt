cmake_minimum_required(VERSION 4.1 FATAL_ERROR)

project(DeeplLearningCuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Specify architectures
# cmake_policy(SET CMP0104 NEW)
set(CMAKE_CUDA_ARCHITECTURES native)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

execute_process(
  COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
  OUTPUT_VARIABLE pybind11_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${pybind11_DIR}")
find_package(pybind11 CONFIG REQUIRED)

# CUDA runtime
find_package(CUDAToolkit REQUIRED)

pybind11_add_module(deep_learning_cuda
  kernels/module.cpp
  kernels/matmul_wrapper.cpp
  kernels/matmul.cu
  kernels/fully_connected_wrapper.cpp
  kernels/fully_connected_launcher.cu
  kernels/fully_connected_fw.cu
  kernels/fully_connected_bw.cu
)

target_link_libraries(deep_learning_cuda PRIVATE CUDA::cudart)
set_target_properties(deep_learning_cuda PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)