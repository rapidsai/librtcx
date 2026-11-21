# SPDX-License-Identifier: Apache-2.0
if(CLANGRTC_FETCH_CUDA_COMPILERS)
  include("${CMAKE_CURRENT_LIST_DIR}/FetchCUDACompilers.cmake")
  clangrtc_fetch_nvjitlink()
endif()
find_package(CUDAToolkit QUIET)
find_path(CLANGRTC_NVJITLINK_INCLUDE_DIR nvJitLink.h
          HINTS ${NVJitLink_ROOT} ENV NVJitLink_ROOT ${CLANGRTC_CUDA_PATH}
                ${CUDAToolkit_INCLUDE_DIRS} PATH_SUFFIXES include)
function(_clangrtc_find_nvjitlink)
  if(CLANGRTC_STATIC_LINK_NVJITLINK)
    set(CMAKE_FIND_LIBRARY_SUFFIXES "${CMAKE_STATIC_LIBRARY_SUFFIX}")
    find_library(CLANGRTC_NVJITLINK_STATIC_LIBRARY NAMES nvJitLink_static
                 HINTS ${NVJitLink_ROOT} ENV NVJitLink_ROOT ${CLANGRTC_CUDA_PATH}
                       ${CUDAToolkit_LIBRARY_ROOT} PATH_SUFFIXES lib lib64)
    get_filename_component(_directory "${CLANGRTC_NVJITLINK_STATIC_LIBRARY}" DIRECTORY)
    find_library(CLANGRTC_NVPTXCOMPILER_STATIC_LIBRARY NAMES nvptxcompiler_static
                 HINTS "${_directory}" ${NVJitLink_ROOT} ENV NVJitLink_ROOT ${CLANGRTC_CUDA_PATH}
                       ${CUDAToolkit_LIBRARY_ROOT} PATH_SUFFIXES lib lib64)
  else()
    set(CMAKE_FIND_LIBRARY_SUFFIXES "${CMAKE_SHARED_LIBRARY_SUFFIX}")
    find_library(CLANGRTC_NVJITLINK_LIBRARY
                 NAMES nvJitLink libnvJitLink.so.13 libnvJitLink.so.12 NAMES_PER_DIR
                 HINTS ${NVJitLink_ROOT} ENV NVJitLink_ROOT ${CLANGRTC_CUDA_PATH}
                       ${CUDAToolkit_LIBRARY_ROOT}
                 PATH_SUFFIXES lib lib64)
  endif()
endfunction()
_clangrtc_find_nvjitlink()
if(CLANGRTC_STATIC_LINK_NVJITLINK)
  set(CLANGRTC_NVJITLINK_SELECTED_LIBRARY "${CLANGRTC_NVJITLINK_STATIC_LIBRARY}")
  set(_nvjitlink_required CLANGRTC_NVPTXCOMPILER_STATIC_LIBRARY)
else()
  set(CLANGRTC_NVJITLINK_SELECTED_LIBRARY "${CLANGRTC_NVJITLINK_LIBRARY}")
endif()
if(CLANGRTC_STATIC_LINK_NVJITLINK)
  foreach(_archive IN ITEMS CLANGRTC_NVJITLINK_STATIC_LIBRARY CLANGRTC_NVPTXCOMPILER_STATIC_LIBRARY)
    if(${_archive} AND NOT EXISTS "${${_archive}}")
      message(FATAL_ERROR "Selected static dependency ${_archive} does not exist: ${${_archive}}")
    endif()
    if(${_archive} AND NOT "${${_archive}}" MATCHES "\\${CMAKE_STATIC_LIBRARY_SUFFIX}$")
      message(FATAL_ERROR "${_archive} must name a static archive")
    endif()
  endforeach()
elseif(CLANGRTC_NVJITLINK_LIBRARY)
  if(NOT EXISTS "${CLANGRTC_NVJITLINK_LIBRARY}")
    message(FATAL_ERROR "Selected nvJitLink library does not exist: ${CLANGRTC_NVJITLINK_LIBRARY}")
  elseif(CLANGRTC_NVJITLINK_LIBRARY MATCHES "\\${CMAKE_STATIC_LIBRARY_SUFFIX}$")
    message(FATAL_ERROR "CLANGRTC_NVJITLINK_LIBRARY must name a shared library; use CLANGRTC_STATIC_LINK_NVJITLINK for archives"
    )
  endif()
endif()
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  NVJitLink REQUIRED_VARS CLANGRTC_NVJITLINK_INCLUDE_DIR CLANGRTC_NVJITLINK_SELECTED_LIBRARY
                          ${_nvjitlink_required})
if(NVJitLink_FOUND AND NOT TARGET NVJitLink::NVJitLink)
  if(CLANGRTC_STATIC_LINK_NVJITLINK)
    find_package(Threads REQUIRED)
    add_library(NVJitLink::NVJitLink STATIC IMPORTED)
    set_property(TARGET NVJitLink::NVJitLink
                 PROPERTY INTERFACE_LINK_LIBRARIES
                          "${CLANGRTC_NVPTXCOMPILER_STATIC_LIBRARY};Threads::Threads;${CMAKE_DL_LIBS}"
    )
  else()
    add_library(NVJitLink::NVJitLink SHARED IMPORTED)
  endif()
  set_property(TARGET NVJitLink::NVJitLink PROPERTY IMPORTED_LOCATION
                                                    "${CLANGRTC_NVJITLINK_SELECTED_LIBRARY}")
  set_property(TARGET NVJitLink::NVJitLink PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                                                    "${CLANGRTC_NVJITLINK_INCLUDE_DIR}")
endif()
