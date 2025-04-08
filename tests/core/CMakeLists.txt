cmake_minimum_required(VERSION 3.15)
project(SparseMoE_Tests CXX)

# Find Google Test package
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.11.0
)
FetchContent_MakeAvailable(googletest)

# Find PyTorch package
find_package(Torch REQUIRED)

# Include directories
include_directories(
  ${CMAKE_SOURCE_DIR}/core/include
  ${TORCH_INCLUDE_DIRS}
)

# Configure CUDA if available
option(USE_CUDA "Use CUDA for GPU acceleration" ON)
if(USE_CUDA)
  find_package(CUDA QUIET)
  if(CUDA_FOUND)
    message(STATUS "CUDA found, building with GPU support")
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
    set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)
    add_definitions(-DUSE_CUDA)
  else()
    message(STATUS "CUDA not found, building CPU-only version")
    set(USE_CUDA OFF)
  endif()
endif()

# Add test executables
add_executable(test_dnps test_dnps.cpp)
target_link_libraries(test_dnps ${TORCH_LIBRARIES} gtest_main)

add_executable(test_attention test_attention.cpp)
target_link_libraries(test_attention ${TORCH_LIBRARIES} gtest_main)

# If CUDA is available, build CUDA-enabled tests
if(USE_CUDA)
  set_property(TARGET test_dnps PROPERTY CUDA_SEPARABLE_COMPILATION ON)
  set_property(TARGET test_attention PROPERTY CUDA_SEPARABLE_COMPILATION ON)
endif()

# Register tests
include(GoogleTest)
gtest_discover_tests(test_dnps)
gtest_discover_tests(test_attention)