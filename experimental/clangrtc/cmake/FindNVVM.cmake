# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Discover pinned or caller-provided libNVVM; LLVM compatibility is checked by the project.
if(CLANGRTC_FETCH_CUDA_COMPILERS)
  include("${CMAKE_CURRENT_LIST_DIR}/FetchCUDACompilers.cmake")
  clangrtc_fetch_nvvm()
endif()
find_package(CUDAToolkit QUIET)
find_path(CLANGRTC_NVVM_INCLUDE_DIR nvvm.h HINTS ${NVVM_ROOT} ENV NVVM_ROOT ${CLANGRTC_CUDA_PATH}
                                                 ${CUDAToolkit_LIBRARY_ROOT}
          PATH_SUFFIXES include nvvm/include)
if(CLANGRTC_STATIC_LINK_NVVM)
  set(_nvvm_names nvvm_static nvvm)
  set(_nvvm_variable CLANGRTC_NVVM_STATIC_LIBRARY)
else()
  set(_nvvm_names nvvm libnvvm.so.4)
  set(_nvvm_variable CLANGRTC_NVVM_LIBRARY)
endif()
# Function scope keeps dependency selection from changing global search suffixes.
function(_clangrtc_find_nvvm)
  if(CLANGRTC_STATIC_LINK_NVVM)
    set(CMAKE_FIND_LIBRARY_SUFFIXES "${CMAKE_STATIC_LIBRARY_SUFFIX}")
  else()
    set(CMAKE_FIND_LIBRARY_SUFFIXES "${CMAKE_SHARED_LIBRARY_SUFFIX}")
  endif()
  find_library(${_nvvm_variable} NAMES ${_nvvm_names} NAMES_PER_DIR
               HINTS ${NVVM_ROOT} ENV NVVM_ROOT ${CLANGRTC_CUDA_PATH} ${CUDAToolkit_LIBRARY_ROOT}
               PATH_SUFFIXES lib lib64 nvvm/lib64)
endfunction()
_clangrtc_find_nvvm()
if(CLANGRTC_STATIC_LINK_NVVM AND NOT CLANGRTC_NVVM_STATIC_LIBRARY)
  message(FATAL_ERROR "Static libNVVM requested, but no archive was found. NVIDIA distributes libNVVM as a shared library only; provide CLANGRTC_NVVM_STATIC_LIBRARY if you have a compatible archive, or disable CLANGRTC_STATIC_LINK_NVVM."
  )
endif()
set(CLANGRTC_NVVM_SELECTED_LIBRARY "${${_nvvm_variable}}")
if(CLANGRTC_NVVM_SELECTED_LIBRARY)
  if(NOT EXISTS "${CLANGRTC_NVVM_SELECTED_LIBRARY}")
    message(FATAL_ERROR "Selected libNVVM library does not exist: ${CLANGRTC_NVVM_SELECTED_LIBRARY}"
    )
  endif()
  if(CLANGRTC_STATIC_LINK_NVVM AND NOT CLANGRTC_NVVM_SELECTED_LIBRARY MATCHES
                                   "\\${CMAKE_STATIC_LIBRARY_SUFFIX}$")
    message(FATAL_ERROR "CLANGRTC_NVVM_STATIC_LIBRARY must name a static archive")
  elseif(NOT CLANGRTC_STATIC_LINK_NVVM AND CLANGRTC_NVVM_SELECTED_LIBRARY MATCHES
                                           "\\${CMAKE_STATIC_LIBRARY_SUFFIX}$")
    message(FATAL_ERROR "CLANGRTC_NVVM_LIBRARY must name a shared library; use CLANGRTC_STATIC_LINK_NVVM for archives"
    )
  endif()
endif()
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NVVM REQUIRED_VARS CLANGRTC_NVVM_INCLUDE_DIR
                                                     CLANGRTC_NVVM_SELECTED_LIBRARY)
if(NVVM_FOUND AND NOT TARGET NVVM::NVVM)
  if(CLANGRTC_STATIC_LINK_NVVM)
    add_library(NVVM::NVVM STATIC IMPORTED)
  else()
    add_library(NVVM::NVVM SHARED IMPORTED)
  endif()
  set_target_properties(NVVM::NVVM
                        PROPERTIES IMPORTED_LOCATION "${CLANGRTC_NVVM_SELECTED_LIBRARY}"
                                   INTERFACE_INCLUDE_DIRECTORIES "${CLANGRTC_NVVM_INCLUDE_DIR}")
endif()
