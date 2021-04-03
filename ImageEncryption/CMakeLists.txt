#设置cmake版本
cmake_minimum_required(VERSION 3.11)
#启用cuda
project(cuda01 LANGUAGES CXX CUDA)

#SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-rdc=true;-std=c++11;)
#设置可执行文件名称
add_executable(imgEncryption main.cu initkey.cu sha256.cpp dna.cu analysis.cu chaos.cu compute.cu mykernel.cu)

find_package(OpenCV REQUIRED)
target_link_libraries(imgEncryption ${OpenCV_LIBS})

list(APPEND CMAKE_CUDA_FLAGS "-rdc=true")