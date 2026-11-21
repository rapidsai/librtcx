# @file tests/curtc/CMakeLists.txt
#
# @brief Build CPU cache tests and integration tests using the real CUDA runtime and GPU.
#
# Discovers CPU cases separately from real loading, ownership, and kernel execution tests.

add_executable(curtc_tests curtc_test.cpp curtc_cache_directory_test.cpp
                           "${CMAKE_CURRENT_SOURCE_DIR}/../support.cpp")
target_compile_features(curtc_tests PRIVATE cxx_std_20)
target_compile_definitions(curtc_tests PRIVATE CURTC_INTERNAL_TESTS
                                               CURTC_BUILD_ID="${_curtc_identity}")
target_include_directories(
    curtc_tests PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/.." "${CMAKE_CURRENT_SOURCE_DIR}/.."
                        "${PROJECT_SOURCE_DIR}/src")
target_include_directories(
    curtc_tests SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS} "${CLANGRTC_CUDA_DRIVER_INCLUDE_DIR}"
                               "${CLANGRTC_NVJITLINK_INCLUDE_DIR}")
target_link_libraries(
    curtc_tests PRIVATE clangrtc::curtc ${_clangrtc_llvm_libraries} xxHash::xxhash ${CMAKE_DL_LIBS}
                        GTest::gtest_main Threads::Threads)
gtest_discover_tests(curtc_tests DISCOVERY_MODE PRE_TEST PROPERTIES LABELS curtc)
if(CURTC_BUILD_GPU_TESTS)
    find_package(CUDAToolkit REQUIRED)
    add_executable(curtc_gpu_tests curtc_gpu_test.cpp curtc_c_test.cpp curtc_c_header.c
                                   gpu_support.cpp "${CMAKE_CURRENT_SOURCE_DIR}/../support.cpp")
    target_compile_features(curtc_gpu_tests PRIVATE cxx_std_20)
    target_include_directories(
        curtc_gpu_tests PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/.." "${CMAKE_CURRENT_SOURCE_DIR}/.."
                                "${PROJECT_SOURCE_DIR}/src")
    target_include_directories(curtc_gpu_tests SYSTEM PRIVATE "${CLANGRTC_NVJITLINK_INCLUDE_DIR}")
    target_link_libraries(
        curtc_gpu_tests PRIVATE clangrtc::curtc GTest::gtest_main ${_clangrtc_cudart}
                                CUDA::cuda_driver NVJitLink::NVJitLink Threads::Threads)
    gtest_discover_tests(curtc_gpu_tests DISCOVERY_MODE PRE_TEST PROPERTIES LABELS gpu
                                                                            RESOURCE_LOCK curtc_gpu)
endif()
