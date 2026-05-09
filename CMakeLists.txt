cmake_minimum_required(VERSION 3.24)

# CUDAFluids builds a realtime 3D CUDA fluid simulation (semi-Lagrangian
# advection, vortex confinement, viscous diffusion, multigrid pressure
# projection, buoyancy, temperature) with a GLFW/OpenGL viewer.
project(CUDAFluids LANGUAGES CXX CUDA)

# Use C++17/CUDA17 so the host and device code can share simple modern C++
# utilities without requiring a newer compiler baseline.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Let CMake choose the native GPU architecture when possible. Users can override
# this with -DCMAKE_CUDA_ARCHITECTURES=<arch> or scripts/build-msvc.ps1
# -CudaArchitectures <arch>.
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures to build for")
endif()

# CUDA supplies the runtime library. OpenGL is used only for the density texture
# display path.
find_package(CUDAToolkit REQUIRED)
find_package(OpenGL REQUIRED)
include(FetchContent)

# Fetch GLFW instead of using an arbitrary system package so MSVC/CUDA builds do
# not accidentally pick up MSYS/MinGW headers.
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.4
)
FetchContent_MakeAvailable(glfw)

# The executable owns both the CUDA solver and the realtime window loop.
add_executable(cuda_fluids
    src/Fluid3D.cu
    src/Multigrid.cu
    src/VolumeRender.cu
    src/main.cpp
)

target_include_directories(cuda_fluids PRIVATE include)
target_link_libraries(cuda_fluids PRIVATE CUDA::cudart OpenGL::GL glfw)

# Separate device linking keeps the CUDA translation unit ready for future
# expansion into multiple CUDA source files.
set_target_properties(cuda_fluids PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
