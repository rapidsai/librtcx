# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

enable_language(ASM)

include(${rapids-cmake-dir}/cpm/gtest.cmake)
rapids_cpm_gtest(BUILD_STATIC)

if(NOT TARGET xxhash)
  set(CPM_DOWNLOAD_xxhash ON)
  rapids_cpm_find(xxhash 0.8.3 CPM_ARGS GIT_REPOSITORY https://github.com/Cyan4973/xxHash.git
                  GIT_TAG v0.8.3 GIT_SHALLOW TRUE DOWNLOAD_ONLY TRUE)
  add_library(xxhash INTERFACE)
  target_include_directories(xxhash INTERFACE "${xxhash_SOURCE_DIR}")
endif()

include(${PROJECT_SOURCE_DIR}/cmake/modules/embed.cmake)

rtcx_add_embed(rtcx_test_embed)
rtcx_embed_includes(rtcx_test_embed SOURCE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include"
                    DEST_DIRECTORY include INCLUDE_DIRECTORIES include)
rtcx_embed(rtcx_test_embed COMPRESSION none
           OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated")

add_executable(rtcx_test rtcx_test.cpp "${rtcx_test_embed_SOURCE_DIR}/rtcx_test_embed.s")
add_dependencies(rtcx_test rtcx_test_embed)
set_target_properties(rtcx_test PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES)
target_include_directories(rtcx_test PRIVATE "${rtcx_test_embed_INCLUDE_DIRS}")
target_link_libraries(rtcx_test PRIVATE rtcx::rtcx CUDA::cudart GTest::gtest_main)

add_test(NAME rtcx_test COMMAND rtcx_test)
