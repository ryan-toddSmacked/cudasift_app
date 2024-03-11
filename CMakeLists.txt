cmake_minimum_required(VERSION 3.0)

project(cudaSift)


find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set cuda architecture
set(CUDA_ARCH "89" CACHE STRING "CUDA architecture")


if (WIN32)
    set(EXTRA_CXX_FLAGS "/DVERBOSE /D_CRT_SECURE_NO_WARNINGS ")
    list(APPEND CUDA_NVCC_FLAGS "-arch=${CUDA_ARCH};--compiler-options;-O2;-DVERBOSE") 
endif()
if (UNIX)
    if (APPLE)
        set(EXTRA_CXX_FLAGS "-DVERBOSE -msse2")
        list(APPEND CUDA_NVCC_FLAGS "-arch=${CUDA_ARCH};--compiler-options;-O2;-DVERBOSE") 
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -msse2 ")
        list(APPEND CUDA_NVCC_FLAGS "-lineinfo;--compiler-options;-O2;-D_FORCE_INLINES;-DVERBOSE_NOT") 
    endif()
endif()

# append fast math flag for cuda
list(APPEND CUDA_NVCC_FLAGS "--use_fast_math")

# All warnings, and treat as errors
if (WIN32)
    set(EXTRA_CXX_FLAGS "${EXTRA_CXX_FLAGS} /W4 /WX")
else()
    set(EXTRA_CXX_FLAGS "${EXTRA_CXX_FLAGS} -Wall -Wextra -Werror -Wpedantic")
endif()

set(cuda_sources
    cudaImage.cu
    cudaImage.h
    cudaSiftH.cu
    cudaSiftH.h
    matching.cu
    cudaSiftD.h
    cudaSift.h
    cudautils.h
)

set(sources
    geomFuncs.cpp
    mainSift.cpp
)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR} 
)

cuda_add_executable(cudasift ${cuda_sources} ${sources} OPTIONS -arch=sm_${CUDA_ARCH})

set_target_properties(cudasift PROPERTIES
    COMPILE_FLAGS "${EXTRA_CXX_FLAGS}"
)

target_link_libraries(cudasift ${CUDA_cudadevrt_LIBRARY} ${OpenCV_LIBS})
 
install(TARGETS cudasift DESTINATION bin)

