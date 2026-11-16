# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

cmake_minimum_required(VERSION 4.0 FATAL_ERROR)

include("${RTCX_REPOSITORY_DIR}/cmake/rapids_config.cmake")
include(rapids-cmake)
include(rapids-cpm)
include(rapids-export)
include(rapids-find)
rapids_cpm_init()

project(rtcx_subdirectory_consumer LANGUAGES CXX)

# Other dependencies built in the same project (e.g. Abseil) define a target named hash.
add_library(hash INTERFACE)

add_subdirectory("${RTCX_REPOSITORY_DIR}" rtcx)
