# SPDX-License-Identifier: Apache-2.0
#
# @file FetchCUDACompilers.cmake
#
# @brief Fetch pinned NVIDIA compiler libraries independently of the application's CUDA toolkit.

include_guard(GLOBAL)
include(FetchContent)

# @brief Populate one checksum-pinned NVIDIA redistributable for the target platform.
function(_clangrtc_fetch_cuda_component component output)

    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
        message(FATAL_ERROR "Automatic CUDA compiler downloads support Linux only. "
                            "Provide dependency roots and disable CLANGRTC_FETCH_CUDA_COMPILERS.")
    endif()

    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
        set(platform linux-x86_64)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
        set(platform linux-sbsa)
    else()
        message(FATAL_ERROR "No pinned CUDA compiler package for ${CMAKE_SYSTEM_PROCESSOR}. "
                            "Provide dependency roots and disable CLANGRTC_FETCH_CUDA_COMPILERS.")
    endif()

    set(manifest_path "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/cuda-compiler-libraries.json")
    set_property(
        DIRECTORY
        APPEND
        PROPERTY CMAKE_CONFIGURE_DEPENDS "${manifest_path}")
    file(READ "${manifest_path}" manifest)
    string(JSON package GET "${manifest}" components "${component}")
    string(JSON version GET "${package}" version)
    string(JSON binary GET "${package}" "${platform}")
    string(JSON archive GET "${binary}" relative_path)
    string(JSON checksum GET "${binary}" sha256)
    set(source "${CLANGRTC_CUDA_COMPILER_CACHE}/${component}-${platform}-${version}")
    # cmake-lint: disable=C0103
    set(FETCHCONTENT_BASE_DIR "${CLANGRTC_CUDA_COMPILER_CACHE}/populate-${platform}")
    FetchContent_Declare(
        clangrtc_${component}
        URL "https://developer.download.nvidia.com/compute/cuda/redist/${archive}"
        URL_HASH "SHA256=${checksum}"
        TLS_VERIFY TRUE
        DOWNLOAD_DIR "${CLANGRTC_CUDA_COMPILER_CACHE}" SOURCE_DIR "${source}"
        DOWNLOAD_EXTRACT_TIMESTAMP FALSE)
    FetchContent_MakeAvailable(clangrtc_${component})
    FetchContent_GetProperties(clangrtc_${component} SOURCE_DIR source)

    if(NOT IS_DIRECTORY "${source}/include" AND NOT IS_DIRECTORY "${source}/nvvm/include")
        message(FATAL_ERROR "Pinned ${component} ${version} was not populated in ${source}. "
                            "Populate the download cache before configuring offline.")
    endif()

    message(STATUS "Using pinned ${component} ${version}: ${source}")
    set(${output}
        "${source}"
        PARENT_SCOPE)
endfunction()

# @brief Preserve caller overrides while clearing cached paths owned by a previous fetched package.
function(_clangrtc_cuda_override root_variable managed_root output)
    set(overridden FALSE)

    foreach(variable IN ITEMS ${root_variable} ${ARGN})
        set(value "${${variable}}")

        if(NOT value)
            continue()
        endif()

        set(managed FALSE)

        if(managed_root)
            cmake_path(IS_PREFIX managed_root "${value}" NORMALIZE managed)
        endif()

        if(managed)
            unset(${variable} CACHE)
        else()
            set(overridden TRUE)
        endif()

    endforeach()

    if(root_variable MATCHES "_ROOT$"
       AND DEFINED ENV{${root_variable}}
       AND NOT "$ENV{${root_variable}}" STREQUAL "")
        set(overridden TRUE)
    endif()

    set(${output}
        "${overridden}"
        PARENT_SCOPE)
endfunction()

# @brief Select the pinned shared libNVVM unless the caller supplies an installation or archive.
function(clangrtc_fetch_nvvm)
    _clangrtc_cuda_override(
        NVVM_ROOT "${_CLANGRTC_FETCHED_NVVM_ROOT}" overridden CLANGRTC_NVVM_INCLUDE_DIR
        CLANGRTC_NVVM_LIBRARY CLANGRTC_NVVM_STATIC_LIBRARY)

    if(overridden OR CLANGRTC_STATIC_LINK_NVVM)
        return()
    endif()

    _clangrtc_fetch_cuda_component(libnvvm root)
    set(NVVM_ROOT
        "${root}"
        PARENT_SCOPE)
    set(CLANGRTC_NVVM_INCLUDE_DIR
        "${root}/nvvm/include"
        PARENT_SCOPE)
    set(CLANGRTC_NVVM_LIBRARY
        "${root}/nvvm/lib64/libnvvm.so"
        PARENT_SCOPE)
    set(CLANGRTC_BUNDLED_NVVM_ROOT
        "${root}"
        PARENT_SCOPE)
    set(_CLANGRTC_FETCHED_NVVM_ROOT
        "${root}"
        CACHE INTERNAL "Previously fetched libNVVM root" FORCE)
endfunction()

# @brief Select pinned nvJitLink and its matching compiler archive when static linkage is requested.
function(clangrtc_fetch_nvjitlink)
    _clangrtc_cuda_override(
        NVJitLink_ROOT "${_CLANGRTC_FETCHED_NVJITLINK_ROOT}" overridden
        CLANGRTC_NVJITLINK_INCLUDE_DIR CLANGRTC_NVJITLINK_LIBRARY CLANGRTC_NVJITLINK_STATIC_LIBRARY)

    if(overridden)
        return()
    endif()

    _clangrtc_fetch_cuda_component(libnvjitlink root)
    set(NVJitLink_ROOT
        "${root}"
        PARENT_SCOPE)
    set(CLANGRTC_NVJITLINK_INCLUDE_DIR
        "${root}/include"
        PARENT_SCOPE)
    if(CLANGRTC_STATIC_LINK_NVJITLINK)
        set(CLANGRTC_NVJITLINK_STATIC_LIBRARY
            "${root}/lib/libnvJitLink_static.a"
            PARENT_SCOPE)
    else()
        set(CLANGRTC_NVJITLINK_LIBRARY
            "${root}/lib/libnvJitLink.so"
            PARENT_SCOPE)
    endif()

    set(CLANGRTC_BUNDLED_NVJITLINK_ROOT
        "${root}"
        PARENT_SCOPE)
    set(_CLANGRTC_FETCHED_NVJITLINK_ROOT
        "${root}"
        CACHE INTERNAL "Previously fetched nvJitLink root" FORCE)

    if(CLANGRTC_STATIC_LINK_NVJITLINK)
        _clangrtc_cuda_override(CLANGRTC_NVPTXCOMPILER_STATIC_LIBRARY
                                "${_CLANGRTC_FETCHED_NVPTXCOMPILER_ROOT}" compiler_overridden)

        if(NOT compiler_overridden)
            _clangrtc_fetch_cuda_component(libnvptxcompiler compiler_root)
            set(CLANGRTC_NVPTXCOMPILER_STATIC_LIBRARY
                "${compiler_root}/lib/libnvptxcompiler_static.a"
                PARENT_SCOPE)
            set(CLANGRTC_BUNDLED_NVPTXCOMPILER_ROOT
                "${compiler_root}"
                PARENT_SCOPE)
            set(_CLANGRTC_FETCHED_NVPTXCOMPILER_ROOT
                "${compiler_root}"
                CACHE INTERNAL "Previously fetched nvPTXCompiler root" FORCE)
        endif()

    endif()

endfunction()

# @brief Obtain private driver headers without linking or installing the package's CUDA runtime.
function(clangrtc_fetch_cuda_driver_headers)
    _clangrtc_cuda_override(CLANGRTC_CUDA_DRIVER_INCLUDE_DIR
                            "${_CLANGRTC_FETCHED_CUDA_DRIVER_ROOT}" overridden)

    if(overridden)
        return()
    endif()

    _clangrtc_fetch_cuda_component(cuda_cudart root)
    set(CLANGRTC_CUDA_DRIVER_INCLUDE_DIR
        "${root}/include"
        PARENT_SCOPE)
    set(_CLANGRTC_FETCHED_CUDA_DRIVER_ROOT
        "${root}"
        CACHE INTERNAL "Previously fetched driver header package" FORCE)
endfunction()

# @brief Install fetched runtime libraries or static-link dependencies alongside clangrtc.
function(clangrtc_install_cuda_compilers)

    foreach(component IN ITEMS NVVM NVJITLINK NVPTXCOMPILER)
        set(root "${CLANGRTC_BUNDLED_${component}_ROOT}")

        if(NOT root)
            continue()
        endif()

        file(GLOB licenses "${root}/LICENSE*" "${root}/nvvm/LICENSE*")
        install(FILES ${licenses}
                DESTINATION "${CMAKE_INSTALL_DATADIR}/clangrtc/licenses/${component}")

        if(component STREQUAL "NVVM")
            file(GLOB libraries "${root}/nvvm/lib64/libnvvm.so*")
            install(FILES ${libraries} DESTINATION "${CMAKE_INSTALL_LIBDIR}")
        elseif(component STREQUAL "NVJITLINK" AND NOT CLANGRTC_STATIC_LINK_NVJITLINK)
            file(GLOB libraries "${root}/lib/libnvJitLink.so*")
            install(FILES ${libraries} DESTINATION "${CMAKE_INSTALL_LIBDIR}")
        elseif(NOT BUILD_SHARED_LIBS)

            if(component STREQUAL "NVJITLINK")
                install(FILES "${root}/lib/libnvJitLink_static.a"
                        DESTINATION "${CMAKE_INSTALL_LIBDIR}")
                install(FILES "${root}/include/nvJitLink.h"
                        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
            else()
                install(FILES "${root}/lib/libnvptxcompiler_static.a"
                        DESTINATION "${CMAKE_INSTALL_LIBDIR}")
            endif()

        endif()

    endforeach()

endfunction()
