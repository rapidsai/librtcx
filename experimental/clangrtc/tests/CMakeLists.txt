# @file tests/CMakeLists.txt
#
# @brief Register compiler and runtime suites with shared test configuration.
#
# Configures common test settings, enables the optional runtime suites, and assigns CTest labels.

find_package(Threads REQUIRED)
include(GoogleTest)
find_path(CLANGRTC_TEST_CCCL_INCLUDE_DIR cuda/std/chrono
          HINTS "${CLANGRTC_CUDA_PATH}/include/cccl"
          DOC "CCCL include root for device standard-library tests")
set(CLANGRTC_TEST_CCCL_FREESTANDING false)
if(EXISTS "${CLANGRTC_TEST_CCCL_INCLUDE_DIR}/cuda/std/__cccl/compiler.h")
  file(READ "${CLANGRTC_TEST_CCCL_INCLUDE_DIR}/cuda/std/__cccl/compiler.h" _cccl_compiler)
  if(_cccl_compiler MATCHES "_CCCL_ENABLE_FREESTANDING")
    set(CLANGRTC_TEST_CCCL_FREESTANDING true)
  endif()
endif()
configure_file(test_config.hpp.in test_config.hpp @ONLY)

add_subdirectory(clangrtc)
option(CURTC_BUILD_GPU_TESTS "Build staged JIT GPU execution tests" OFF)
if(CLANGRTC_BUILD_CURTC)
  add_subdirectory(curtc)
endif()
