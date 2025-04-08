cmake_minimum_required(VERSION 3.22)
project(SparseMoE LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Set output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Find dependencies
find_package(CUDA 12.0 REQUIRED)
find_package(Torch REQUIRED)
find_package(pybind11 REQUIRED)

# Enable CUDA support
include_directories(${CUDA_INCLUDE_DIRS})
set(CMAKE_CUDA_ARCHITECTURES 80 86 90)

# Set include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${TORCH_INCLUDE_DIRS}
)

# Add source files by component
file(GLOB_RECURSE ATTENTION_SOURCES "src/attention/*.cpp")
file(GLOB_RECURSE DNPS_SOURCES "src/dnps/*.cpp")
file(GLOB_RECURSE DISTRIBUTED_SOURCES "src/distributed/*.cpp")
file(GLOB_RECURSE QUANTIZATION_SOURCES "src/quantization/*.cpp")
file(GLOB_RECURSE CUDA_SOURCES "src/cuda/*.cu")

# Create core library
add_library(sparsemoe_core SHARED
    ${ATTENTION_SOURCES}
    ${DNPS_SOURCES}
    ${DISTRIBUTED_SOURCES}
    ${QUANTIZATION_SOURCES}
    ${CUDA_SOURCES}
)

target_link_libraries(sparsemoe_core PUBLIC ${TORCH_LIBRARIES} ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})

# Create Python bindings
pybind11_add_module(sparsemoe_bindings bindings/bindings.cpp)
target_link_libraries(sparsemoe_bindings PRIVATE sparsemoe_core)

# Create standalone executables
add_executable(sparsemoe-generate bindings/apps/generate.cpp)
target_link_libraries(sparsemoe-generate PRIVATE sparsemoe_core)

add_executable(sparsemoe-train bindings/apps/train.cpp)
target_link_libraries(sparsemoe-train PRIVATE sparsemoe_core)

add_executable(sparsemoe-convert bindings/apps/convert.cpp)
target_link_libraries(sparsemoe-convert PRIVATE sparsemoe_core)

# Install rules
install(TARGETS sparsemoe_core sparsemoe_bindings sparsemoe-generate sparsemoe-train sparsemoe-convert
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# Install headers
install(DIRECTORY include/ DESTINATION include/sparsemoe)