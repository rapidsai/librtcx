# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

cmake_minimum_required(VERSION 4.0 FATAL_ERROR)

project(rtcx_cmake_tests LANGUAGES NONE)

enable_testing()

set(RTCX_REPOSITORY_DIR "" CACHE PATH "Path to the librtcx repository root")
if(NOT RTCX_REPOSITORY_DIR)
  cmake_path(SET RTCX_REPOSITORY_DIR NORMALIZE "${CMAKE_CURRENT_LIST_DIR}/../..")
endif()

set(rtcx_build_dir "${CMAKE_CURRENT_BINARY_DIR}/rtcx-build")
set(rtcx_install_dir "${CMAKE_CURRENT_BINARY_DIR}/rtcx-install")
set(consumer_source_dir "${CMAKE_CURRENT_LIST_DIR}/consumer")

add_test(NAME rtcx_configure COMMAND ${CMAKE_COMMAND} -S "${RTCX_REPOSITORY_DIR}" -B
                                     "${rtcx_build_dir}" -G "${CMAKE_GENERATOR}")
add_test(NAME rtcx_build COMMAND ${CMAKE_COMMAND} --build "${rtcx_build_dir}")
add_test(NAME rtcx_install COMMAND ${CMAKE_COMMAND} --install "${rtcx_build_dir}" --prefix
                                   "${rtcx_install_dir}")
set_tests_properties(rtcx_configure PROPERTIES FIXTURES_SETUP rtcx_configured)
set_tests_properties(rtcx_build PROPERTIES FIXTURES_REQUIRED rtcx_configured FIXTURES_SETUP
                                           rtcx_built)
set_tests_properties(rtcx_install PROPERTIES FIXTURES_REQUIRED rtcx_built FIXTURES_SETUP
                                             rtcx_installed)

# Build the consumer against both the build tree and the install tree.
foreach(variant IN ITEMS build install)
  set(consumer_build_dir "${CMAKE_CURRENT_BINARY_DIR}/consumer-${variant}")
  if(variant STREQUAL "build")
    set(rtcx_location "-Drtcx_DIR=${rtcx_build_dir}")
    set(fixture rtcx_built)
  else()
    set(rtcx_location "-Drtcx_ROOT=${rtcx_install_dir}")
    set(fixture rtcx_installed)
  endif()

  add_test(NAME consumer_${variant}_configure
           COMMAND ${CMAKE_COMMAND} -S "${consumer_source_dir}" -B "${consumer_build_dir}" -G
                   "${CMAKE_GENERATOR}" "${rtcx_location}")
  add_test(NAME consumer_${variant}_build COMMAND ${CMAKE_COMMAND} --build "${consumer_build_dir}")
  set_tests_properties(consumer_${variant}_configure
                       PROPERTIES FIXTURES_REQUIRED ${fixture} FIXTURES_SETUP
                                  consumer_${variant}_configured)
  add_test(NAME consumer_${variant}_run COMMAND "${consumer_build_dir}/consumer")
  set_tests_properties(consumer_${variant}_build
                       PROPERTIES FIXTURES_REQUIRED consumer_${variant}_configured FIXTURES_SETUP
                                  consumer_${variant}_built)
  set_tests_properties(consumer_${variant}_run PROPERTIES FIXTURES_REQUIRED
                                                          consumer_${variant}_built)
endforeach()
