# SPDX-License-Identifier: Apache-2.0
#
# @file FindxxHash.cmake
#
# @brief Select Cyan's xxHash library, requiring runtime dispatch on x86.

find_path(
    CLANGRTC_XXHASH_INCLUDE_DIR xxhash.h
    HINTS ${xxHash_ROOT} ENV xxHash_ROOT
    PATH_SUFFIXES include)

# @brief Restrict library selection to the caller's requested dependency linkage.
# CMAKE_FIND_LIBRARY_SUFFIXES is a standard CMake variable scoped to this function.
#
# cmake-lint: disable=C0103
function(_clangrtc_find_xxhash)
    if(CLANGRTC_STATIC_LINK_XXHASH)
        set(CMAKE_FIND_LIBRARY_SUFFIXES "${CMAKE_STATIC_LIBRARY_SUFFIX}")
        find_library(
            CLANGRTC_XXHASH_STATIC_LIBRARY
            NAMES xxhash
            HINTS ${xxHash_ROOT} ENV xxHash_ROOT
            PATH_SUFFIXES lib lib64)
    else()
        set(CMAKE_FIND_LIBRARY_SUFFIXES "${CMAKE_SHARED_LIBRARY_SUFFIX}")
        find_library(
            CLANGRTC_XXHASH_LIBRARY
            NAMES xxhash libxxhash.so.0
            HINTS ${xxHash_ROOT} ENV xxHash_ROOT
            PATH_SUFFIXES lib lib64)
    endif()
endfunction()
_clangrtc_find_xxhash()
if(CLANGRTC_STATIC_LINK_XXHASH)
    set(CLANGRTC_XXHASH_SELECTED_LIBRARY "${CLANGRTC_XXHASH_STATIC_LIBRARY}")
    set(_xxhash_type STATIC)
else()
    set(CLANGRTC_XXHASH_SELECTED_LIBRARY "${CLANGRTC_XXHASH_LIBRARY}")
    set(_xxhash_type SHARED)
endif()

if(CLANGRTC_STATIC_LINK_XXHASH
   AND CLANGRTC_XXHASH_SELECTED_LIBRARY
   AND NOT CLANGRTC_XXHASH_SELECTED_LIBRARY MATCHES "\\${CMAKE_STATIC_LIBRARY_SUFFIX}$")
    message(FATAL_ERROR "CLANGRTC_XXHASH_STATIC_LIBRARY must name a static archive")
elseif(NOT CLANGRTC_STATIC_LINK_XXHASH AND CLANGRTC_XXHASH_SELECTED_LIBRARY MATCHES
                                           "\\${CMAKE_STATIC_LIBRARY_SUFFIX}$")
    message(FATAL_ERROR "Use CLANGRTC_STATIC_LINK_XXHASH=ON to select a static xxHash archive")
endif()

if(CLANGRTC_XXHASH_INCLUDE_DIR AND CLANGRTC_XXHASH_SELECTED_LIBRARY)
    include(CheckCXXSourceCompiles)
    set(CMAKE_REQUIRED_INCLUDES "${CLANGRTC_XXHASH_INCLUDE_DIR}")
    set(CMAKE_REQUIRED_LIBRARIES "${CLANGRTC_XXHASH_SELECTED_LIBRARY}")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|i.86)$")
        set(_xxhash_header xxh_x86dispatch.h)
    else()
        set(_xxhash_header xxhash.h)
    endif()
    unset(_xxhash_works CACHE)
    check_cxx_source_compiles(
        "#include <${_xxhash_header}>\nint main() { return int(XXH3_128bits(nullptr, 0).low64); }"
        _xxhash_works)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    xxHash
    REQUIRED_VARS
        CLANGRTC_XXHASH_INCLUDE_DIR CLANGRTC_XXHASH_SELECTED_LIBRARY _xxhash_works
        REASON_FAILURE_MESSAGE
        "On x86, build Cyan xxHash with DISPATCH=ON, including for static archives.")
if(xxHash_FOUND AND NOT TARGET xxHash::xxhash)
    add_library(xxHash::xxhash ${_xxhash_type} IMPORTED)
    set_target_properties(
        xxHash::xxhash PROPERTIES IMPORTED_LOCATION "${CLANGRTC_XXHASH_SELECTED_LIBRARY}"
                                  INTERFACE_INCLUDE_DIRECTORIES "${CLANGRTC_XXHASH_INCLUDE_DIR}")
endif()
