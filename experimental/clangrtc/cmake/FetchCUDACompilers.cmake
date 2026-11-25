# SPDX-License-Identifier: Apache-2.0
#
# @file FetchCUDACompilers.cmake
#
# @brief Fetch pinned NVIDIA compiler libraries independently of the application's CUDA toolkit.

include_guard(GLOBAL)
include(FetchContent)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "Pinned CUDA compiler packages support Linux only.")
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
    set(_clangrtc_cuda_platform linux-x86_64)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
    set(_clangrtc_cuda_platform linux-sbsa)
else()
    message(FATAL_ERROR "No pinned CUDA compiler package for ${CMAKE_SYSTEM_PROCESSOR}.")
endif()

set(_clangrtc_cuda_manifest_path "${CMAKE_CURRENT_LIST_DIR}/cuda-compiler-libraries.json")
set_property(
    DIRECTORY
    APPEND
    PROPERTY CMAKE_CONFIGURE_DEPENDS "${_clangrtc_cuda_manifest_path}")
file(READ "${_clangrtc_cuda_manifest_path}" _clangrtc_cuda_manifest)

# @brief Populate one checksum-pinned NVIDIA redistributable for the target platform.
function(_clangrtc_fetch_cuda_component component output)
    string(JSON package GET "${_clangrtc_cuda_manifest}" components "${component}")
    string(JSON version GET "${package}" version)
    string(JSON binary GET "${package}" "${_clangrtc_cuda_platform}")
    string(JSON archive GET "${binary}" relative_path)
    string(JSON checksum GET "${binary}" sha256)
    set(source "${CLANGRTC_CUDA_COMPILER_CACHE}/${component}-${_clangrtc_cuda_platform}-${version}")
    # cmake-lint: disable=C0103
    set(FETCHCONTENT_BASE_DIR "${CLANGRTC_CUDA_COMPILER_CACHE}/populate-${_clangrtc_cuda_platform}")
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

# @brief Select the pinned shared libNVVM package for the target platform.
function(clangrtc_fetch_nvvm)
    _clangrtc_fetch_cuda_component(libnvvm root)
    set(CLANGRTC_NVVM_INCLUDE_DIR
        "${root}/nvvm/include"
        PARENT_SCOPE)
    set(CLANGRTC_NVVM_SELECTED_LIBRARY
        "${root}/nvvm/lib64/libnvvm.so"
        PARENT_SCOPE)
    set(CLANGRTC_BUNDLED_NVVM_ROOT
        "${root}"
        PARENT_SCOPE)
endfunction()

# @brief Select pinned nvJitLink and its matching compiler archive when static linkage is requested.
function(clangrtc_fetch_nvjitlink)
    _clangrtc_fetch_cuda_component(libnvjitlink root)
    set(CLANGRTC_NVJITLINK_INCLUDE_DIR
        "${root}/include"
        PARENT_SCOPE)
    set(CLANGRTC_BUNDLED_NVJITLINK_ROOT
        "${root}"
        PARENT_SCOPE)

    if(CLANGRTC_STATIC_LINK_NVJITLINK)
        _clangrtc_fetch_cuda_component(libnvptxcompiler compiler_root)
        set(compiler_library "${compiler_root}/lib/libnvptxcompiler_static.a")
        set(CLANGRTC_NVPTXCOMPILER_STATIC_LIBRARY
            "${compiler_library}"
            PARENT_SCOPE)
        set(CLANGRTC_BUNDLED_NVPTXCOMPILER_ROOT
            "${compiler_root}"
            PARENT_SCOPE)

        add_library(clangrtc::nvPTXCompiler STATIC IMPORTED)
        set_target_properties(clangrtc::nvPTXCompiler PROPERTIES IMPORTED_LOCATION
                                                                 "${compiler_library}")
        find_package(Threads REQUIRED)
        add_library(clangrtc::nvJitLink STATIC IMPORTED)
        set_target_properties(
            clangrtc::nvJitLink
            PROPERTIES INTERFACE_LINK_LIBRARIES
                       "clangrtc::nvPTXCompiler;Threads::Threads;${CMAKE_DL_LIBS}")
        set(library "${root}/lib/libnvJitLink_static.a")
    else()
        add_library(clangrtc::nvJitLink SHARED IMPORTED)
        set(library "${root}/lib/libnvJitLink.so")
    endif()

    set_target_properties(
        clangrtc::nvJitLink PROPERTIES IMPORTED_LOCATION "${library}" INTERFACE_INCLUDE_DIRECTORIES
                                                                      "${root}/include")
    set(CLANGRTC_NVJITLINK_SELECTED_LIBRARY
        "${library}"
        PARENT_SCOPE)

endfunction()

# @brief Obtain private driver headers without linking or installing the package's CUDA runtime.
function(clangrtc_fetch_cuda_driver_headers)
    _clangrtc_fetch_cuda_component(cuda_cudart root)
    set(CLANGRTC_CUDA_DRIVER_INCLUDE_DIR
        "${root}/include"
        PARENT_SCOPE)
endfunction()

# @brief Install the fetched backend libraries and their licenses for the selected linkage.
function(clangrtc_install_cuda_compilers)
    foreach(component IN ITEMS NVVM NVJITLINK NVPTXCOMPILER)
        set(root "${CLANGRTC_BUNDLED_${component}_ROOT}")

        if(root)
            file(GLOB licenses "${root}/LICENSE*" "${root}/nvvm/LICENSE*")
            install(FILES ${licenses}
                    DESTINATION "${CMAKE_INSTALL_DATADIR}/clangrtc/licenses/${component}")
        endif()
    endforeach()

    file(GLOB nvvm_libraries "${CLANGRTC_BUNDLED_NVVM_ROOT}/nvvm/lib64/libnvvm.so*")
    install(FILES ${nvvm_libraries} DESTINATION "${CMAKE_INSTALL_LIBDIR}")

    if(CLANGRTC_BUILD_CURTC)
        if(CLANGRTC_STATIC_LINK_NVJITLINK)
            if(NOT BUILD_SHARED_LIBS)
                install(FILES "${CLANGRTC_BUNDLED_NVJITLINK_ROOT}/lib/libnvJitLink_static.a"
                              "${CLANGRTC_NVPTXCOMPILER_STATIC_LIBRARY}"
                        DESTINATION "${CMAKE_INSTALL_LIBDIR}")
            endif()
        else()
            file(GLOB nvjitlink_libraries "${CLANGRTC_BUNDLED_NVJITLINK_ROOT}/lib/libnvJitLink.so*")
            install(FILES ${nvjitlink_libraries} DESTINATION "${CMAKE_INSTALL_LIBDIR}")
        endif()
    endif()
endfunction()
