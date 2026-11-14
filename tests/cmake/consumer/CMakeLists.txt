# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

cmake_minimum_required(VERSION 4.0 FATAL_ERROR)

project(rtcx_consumer LANGUAGES CXX ASM)

find_package(rtcx CONFIG REQUIRED)

# rtcx_embed() requires the consumer to provide zstd and xxhash targets.
add_library(zstd ALIAS zstd::libzstd_static)
find_path(xxhash_INCLUDE_DIR xxhash.h REQUIRED)
add_library(xxhash INTERFACE)
target_include_directories(xxhash INTERFACE "${xxhash_INCLUDE_DIR}")

rtcx_add_embed(consumer_embed)
rtcx_embed_includes(consumer_embed SOURCE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include"
                    DEST_DIRECTORY include INCLUDE_DIRECTORIES include)
rtcx_embed_includes(consumer_embed SOURCE_DIRECTORY "${rtcx_LIBCXX_DIR}" DEST_DIRECTORY rtcx/libcxx
                    INCLUDE_DIRECTORIES rtcx/libcxx)
rtcx_embed(consumer_embed COMPRESSION zstd OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated")

add_executable(consumer main.cpp "${consumer_embed_SOURCE_DIR}/consumer_embed.s")
add_dependencies(consumer consumer_embed)
set_target_properties(consumer PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES)
target_include_directories(consumer PRIVATE "${consumer_embed_INCLUDE_DIRS}")
target_link_libraries(consumer PRIVATE rtcx::rtcx)
