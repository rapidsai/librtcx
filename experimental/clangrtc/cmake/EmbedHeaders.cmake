# SPDX-License-Identifier: Apache-2.0
#
# @file EmbedHeaders.cmake @brief Bundle clangrtc workarounds and Clang resource headers using
# rtcx_embed.

set(_rtcx_source "${CMAKE_CURRENT_SOURCE_DIR}/../..")
include("${_rtcx_source}/cmake/modules/embed.cmake")
enable_language(ASM)

# These dependencies belong to the build-time generator, not the installed libraries.
if(NOT TARGET rtcx::hash)
    add_library(clangrtc_embed_hash STATIC "${_rtcx_source}/src/hash.cpp")
    target_compile_features(clangrtc_embed_hash PRIVATE cxx_std_20)
    target_include_directories(clangrtc_embed_hash PUBLIC "${_rtcx_source}/include")
    add_library(rtcx::hash ALIAS clangrtc_embed_hash)
endif()
if(NOT TARGET xxhash)
    add_library(xxhash INTERFACE)
    target_include_directories(xxhash INTERFACE "${clangrtc_xxhash_SOURCE_DIR}")
endif()
if(NOT TARGET zstd)
    find_package(zstd CONFIG REQUIRED)
    if(LLVM_USE_STATIC_ZSTD)
        add_library(zstd ALIAS zstd::libzstd_static)
    else()
        add_library(zstd ALIAS zstd::libzstd_shared)
    endif()
endif()

set(_clangrtc_resource_path "/__clangrtc__/resource")
rtcx_add_embed(clangrtc_headers)
rtcx_embed_includes(
    clangrtc_headers
    SOURCE_DIRECTORY
    "${CMAKE_CURRENT_SOURCE_DIR}/headers"
    DEST_DIRECTORY
    /__clangrtc__/include
    INCLUDE_DIRECTORIES
    /__clangrtc__/include
    FILES
    clangrtc/device.hpp
    clangrtc/libcudacxx/algorithm
    clangrtc/libcudacxx/array
    clangrtc/libcudacxx/bit
    clangrtc/libcudacxx/cassert
    clangrtc/libcudacxx/chrono
    clangrtc/libcudacxx/climits
    clangrtc/libcudacxx/cmath
    clangrtc/libcudacxx/config.hpp
    clangrtc/libcudacxx/cstddef
    clangrtc/libcudacxx/cstdint
    clangrtc/libcudacxx/cstdlib
    clangrtc/libcudacxx/functional
    clangrtc/libcudacxx/initializer_list
    clangrtc/libcudacxx/iterator
    clangrtc/libcudacxx/limits
    clangrtc/libcudacxx/new
    clangrtc/libcudacxx/optional
    clangrtc/libcudacxx/ratio
    clangrtc/libcudacxx/span
    clangrtc/libcudacxx/string_view
    clangrtc/libcudacxx/tuple
    clangrtc/libcudacxx/type_traits
    clangrtc/libcudacxx/utility)
rtcx_embed_includes(
    clangrtc_headers
    SOURCE_DIRECTORY
    "${CLANGRTC_RESOURCE_DIR}/include"
    DEST_DIRECTORY
    "${_clangrtc_resource_path}/include"
    INCLUDE_DIRECTORIES
    "${_clangrtc_resource_path}/include"
    # Review this CUDA/NVPTX and portable C inventory when updating LLVM.
    FILES
    # CUDA device support, including generic GPU intrinsics on NVPTX.
    __clang_cuda_builtin_vars.h
    __clang_cuda_cmath.h
    __clang_cuda_complex_builtins.h
    __clang_cuda_device_functions.h
    __clang_cuda_intrinsics.h
    __clang_cuda_libdevice_declares.h
    __clang_cuda_math.h
    __clang_cuda_math_forward_declares.h
    __clang_cuda_runtime_wrapper.h
    __clang_cuda_texture_intrinsics.h
    __clang_gpu_builtin_vars.h
    __clang_gpu_device_functions.h
    __clang_gpu_intrinsics.h
    gpuintrin.h
    nvptxintrin.h
    # CUDA adapters for the external host C++ standard library.
    cuda_wrappers/__utility/declval.h
    cuda_wrappers/algorithm
    cuda_wrappers/bits/basic_string.h
    cuda_wrappers/bits/basic_string.tcc
    cuda_wrappers/bits/c++config.h
    cuda_wrappers/bits/shared_ptr_base.h
    cuda_wrappers/cmath
    cuda_wrappers/complex
    cuda_wrappers/new
    # Portable C definitions shared by CUDA and native host compilation.
    __float_float.h
    __float_header_macro.h
    __float_infinity_nan.h
    __stdarg___gnuc_va_list.h
    __stdarg___va_copy.h
    __stdarg_header_macro.h
    __stdarg_va_arg.h
    __stdarg_va_copy.h
    __stdarg_va_list.h
    __stddef_header_macro.h
    __stddef_max_align_t.h
    __stddef_null.h
    __stddef_nullptr_t.h
    __stddef_offsetof.h
    __stddef_ptrdiff_t.h
    __stddef_rsize_t.h
    __stddef_size_t.h
    __stddef_unreachable.h
    __stddef_wchar_t.h
    __stddef_wint_t.h
    float.h
    inttypes.h
    iso646.h
    limits.h
    stdalign.h
    stdarg.h
    stdatomic.h
    stdbool.h
    stdckdint.h
    stdcountof.h
    stddef.h
    stddefer.h
    stdint.h
    stdnoreturn.h
    tgmath.h
    unwind.h)

rtcx_embed(clangrtc_headers COMPRESSION none OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
# Use the same pinned xxHash for the generator even when embedded in a parent project.
target_include_directories(clangrtc_headers__jit_embed_run BEFORE
                           PRIVATE "${clangrtc_xxhash_SOURCE_DIR}")
get_property(
    _clangrtc_embedded_sources
    TARGET clangrtc_headers__embed_props
    PROPERTY EMBED_SOURCE_FILES)
