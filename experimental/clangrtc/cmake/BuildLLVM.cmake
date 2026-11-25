# SPDX-License-Identifier: Apache-2.0

# Build LLVM in its own project so its build targets do not enter this project.
include(ExternalProject)
# The supported LLVM release and its published source checksum are build inputs, not user options.
set(CLANGRTC_LLVM_VERSION "23.1.5")
set(CLANGRTC_LLVM_SHA256 "fb50b1eac93ddeb3aa8dab4cfe324dff74f4a9c2bdee6160c9e36188f8d06a6e")
set(CLANGRTC_LLVM_SOURCE_DIR
    ""
    CACHE PATH "Existing llvm-project source tree; otherwise download the pinned release")
set(CLANGRTC_LLVM_CACHE
    "${CMAKE_CURRENT_BINARY_DIR}/llvm-downloads"
    CACHE PATH "Private LLVM build and download cache")

set(_llvm_source "${CLANGRTC_LLVM_SOURCE_DIR}")
set(_llvm_download)

if(_llvm_source)
    if(NOT EXISTS "${_llvm_source}/llvm/CMakeLists.txt")
        message(FATAL_ERROR "CLANGRTC_LLVM_SOURCE_DIR must name an llvm-project source tree")
    endif()

    include("${_llvm_source}/cmake/Modules/LLVMVersion.cmake")
    set(_source_version "${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}.${LLVM_VERSION_PATCH}")

    if(NOT _source_version VERSION_EQUAL CLANGRTC_LLVM_VERSION)
        message(
            FATAL_ERROR
                "LLVM source version ${_source_version} does not match pinned version ${CLANGRTC_LLVM_VERSION}"
        )
    endif()
else()
    set(_llvm_source "${CLANGRTC_LLVM_CACHE}/llvm-project-${CLANGRTC_LLVM_VERSION}.src")
    set(_llvm_download
        URL
        "https://github.com/llvm/llvm-project/releases/download/llvmorg-${CLANGRTC_LLVM_VERSION}/llvm-project-${CLANGRTC_LLVM_VERSION}.src.tar.xz"
        URL_HASH
        "SHA256=${CLANGRTC_LLVM_SHA256}"
        DOWNLOAD_DIR
        "${CLANGRTC_LLVM_CACHE}"
        DOWNLOAD_EXTRACT_TIMESTAMP
        FALSE
        TLS_VERIFY
        ON)
endif()

set(_llvm_install "${CLANGRTC_LLVM_CACHE}/install-${CLANGRTC_LLVM_VERSION}")
string(REGEX MATCH "^[0-9]+" _llvm_major "${CLANGRTC_LLVM_VERSION}")
# @brief Encode explicitly supported settings, leaving undefined values to the child project's
# defaults.
function(_clangrtc_cmake_args output)
    set(_arguments)

    foreach(_name IN LISTS ARGN)
        if(DEFINED ${_name})
            string(REPLACE ";" "$<SEMICOLON>" _value "${${_name}}")
            list(APPEND _arguments "-D${_name}=${_value}")
        endif()
    endforeach()

    set(${output}
        "${_arguments}"
        PARENT_SCOPE)
endfunction()

# Compilers, toolchains, flags, and system dependencies shared by both builds.
_clangrtc_cmake_args(
    _common_args
    CMAKE_C_COMPILER
    CMAKE_CXX_COMPILER
    CMAKE_ASM_COMPILER
    CMAKE_C_COMPILER_LAUNCHER
    CMAKE_CXX_COMPILER_LAUNCHER
    CMAKE_C_COMPILER_TARGET
    CMAKE_CXX_COMPILER_TARGET
    CMAKE_ASM_COMPILER_TARGET
    CMAKE_C_COMPILER_EXTERNAL_TOOLCHAIN
    CMAKE_CXX_COMPILER_EXTERNAL_TOOLCHAIN
    CMAKE_ASM_COMPILER_EXTERNAL_TOOLCHAIN
    CMAKE_CROSSCOMPILING_EMULATOR
    CMAKE_TOOLCHAIN_FILE
    CMAKE_SYSROOT
    CMAKE_SYSROOT_COMPILE
    CMAKE_SYSROOT_LINK
    CMAKE_PREFIX_PATH
    CMAKE_FIND_ROOT_PATH
    CMAKE_FIND_ROOT_PATH_MODE_PROGRAM
    CMAKE_FIND_ROOT_PATH_MODE_LIBRARY
    CMAKE_FIND_ROOT_PATH_MODE_INCLUDE
    CMAKE_FIND_ROOT_PATH_MODE_PACKAGE
    CMAKE_C_FLAGS
    CMAKE_CXX_FLAGS
    CMAKE_ASM_FLAGS
    CMAKE_EXE_LINKER_FLAGS
    CMAKE_SHARED_LINKER_FLAGS
    CMAKE_MODULE_LINKER_FLAGS
    CMAKE_STATIC_LINKER_FLAGS
    CMAKE_CONFIGURATION_TYPES
    ZLIB_ROOT
    ZLIB_USE_STATIC_LIBS
    ZLIB_INCLUDE_DIR
    ZLIB_LIBRARY
    ZLIB_LIBRARY_RELEASE
    ZLIB_LIBRARY_DEBUG
    zstd_ROOT
    zstd_DIR
    LibXml2_ROOT
    LibXml2_DIR
    LIBXML2_INCLUDE_DIR
    LIBXML2_LIBRARY)

foreach(_configuration IN ITEMS DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    _clangrtc_cmake_args(
        _configuration_args
        CMAKE_C_FLAGS_${_configuration}
        CMAKE_CXX_FLAGS_${_configuration}
        CMAKE_ASM_FLAGS_${_configuration}
        CMAKE_EXE_LINKER_FLAGS_${_configuration}
        CMAKE_SHARED_LINKER_FLAGS_${_configuration}
        CMAKE_MODULE_LINKER_FLAGS_${_configuration}
        CMAKE_STATIC_LINKER_FLAGS_${_configuration})
    list(APPEND _common_args ${_configuration_args})
endforeach()

# Native builds derive their system identity; cross builds must preserve the selected target.
if(CMAKE_CROSSCOMPILING)
    _clangrtc_cmake_args(_target_args CMAKE_SYSTEM_NAME CMAKE_SYSTEM_VERSION CMAKE_SYSTEM_PROCESSOR)
    list(APPEND _common_args ${_target_args})
endif()

_clangrtc_cmake_args(_llvm_args LLVM_ENABLE_ZLIB LLVM_ENABLE_ZSTD LLVM_ENABLE_LIBXML2
                     LLVM_USE_STATIC_ZSTD)

# Public build switches, dependency overrides, and install destinations for clangrtc/curtc.
# LLVM_DIR, Clang_DIR, and the resource directory always come from our private installation.
_clangrtc_cmake_args(
    _inner_args
    BUILD_SHARED_LIBS
    BUILD_TESTING
    CMAKE_BUILD_TYPE
    CLANGRTC_BUILD_CURTC
    CLANGRTC_BUILD_DOCS
    CURTC_BUILD_GPU_TESTS
    CLANGRTC_STATIC_LINK_NVJITLINK
    CLANGRTC_STATIC_LINK_CUDART
    CLANGRTC_CUDA_COMPILER_CACHE
    CLANGRTC_CUDA_PATH
    CLANGRTC_TEST_CCCL_INCLUDE_DIR
    CUDAToolkit_ROOT
    GTest_ROOT
    GTest_DIR
    DOXYGEN_EXECUTABLE
    FETCHCONTENT_FULLY_DISCONNECTED
    FETCHCONTENT_UPDATES_DISCONNECTED
    FETCHCONTENT_QUIET
    FETCHCONTENT_SOURCE_DIR_CLANGRTC_LIBNVVM
    FETCHCONTENT_SOURCE_DIR_CLANGRTC_LIBNVJITLINK
    FETCHCONTENT_SOURCE_DIR_CLANGRTC_LIBNVPTXCOMPILER
    FETCHCONTENT_SOURCE_DIR_CLANGRTC_CUDA_CUDART
    CMAKE_INSTALL_PREFIX
    CMAKE_INSTALL_LIBDIR
    CMAKE_INSTALL_INCLUDEDIR
    CMAKE_INSTALL_BINDIR
    CMAKE_INSTALL_DATAROOTDIR
    CMAKE_INSTALL_DATADIR)

ExternalProject_Add(
    clangrtc_llvm
    ${_llvm_download}
    SOURCE_DIR "${_llvm_source}"
    SOURCE_SUBDIR llvm
    BINARY_DIR "${CLANGRTC_LLVM_CACHE}/build-${CLANGRTC_LLVM_VERSION}"
    CMAKE_ARGS ${_common_args}
               ${_llvm_args}
               -DCMAKE_BUILD_TYPE=Release
               -DLLVM_ENABLE_PROJECTS=clang
               "-DLLVM_TARGETS_TO_BUILD=Native$<SEMICOLON>NVPTX"
               -DBUILD_SHARED_LIBS=OFF
               -DLLVM_BUILD_LLVM_DYLIB=OFF
               -DLLVM_LINK_LLVM_DYLIB=OFF
               -DCLANG_LINK_CLANG_DYLIB=OFF
               -DCMAKE_POSITION_INDEPENDENT_CODE=ON
               -DLLVM_INCLUDE_TESTS=OFF
               -DLLVM_INCLUDE_EXAMPLES=OFF
               -DLLVM_INCLUDE_BENCHMARKS=OFF
               -DLLVM_BUILD_TOOLS=OFF
               -DCLANG_BUILD_TOOLS=OFF
               -DLLVM_ENABLE_BINDINGS=OFF
               -DLLVM_ENABLE_LIBEDIT=OFF
               "-DCMAKE_INSTALL_PREFIX=${_llvm_install}")

set(_inner_build "${CMAKE_CURRENT_BINARY_DIR}/clangrtc-build")
ExternalProject_Add(
    clangrtc_libraries
    DEPENDS clangrtc_llvm
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
    BINARY_DIR "${_inner_build}"
    CMAKE_ARGS ${_common_args}
               ${_inner_args}
               -DCLANGRTC_FETCH_LLVM=OFF
               "-DLLVM_DIR=${_llvm_install}/lib/cmake/llvm"
               "-DClang_DIR=${_llvm_install}/lib/cmake/clang"
               "-DCLANGRTC_RESOURCE_DIR=${_llvm_install}/lib/clang/${_llvm_major}"
    BUILD_ALWAYS ON
    INSTALL_COMMAND "")

# Let ctest discover the inner project's individual tests after it is built.
option(BUILD_TESTING "Build tests" ON)
if(BUILD_TESTING)
    file(
        GENERATE
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/CTestTestfile.cmake"
        CONTENT "subdirs(\"${_inner_build}\")\n")
endif()
set(_install_inner
    [=[
execute_process(COMMAND "@CMAKE_COMMAND@" --install "@_inner_build@" --prefix
                        "${CMAKE_INSTALL_PREFIX}" --config "${CMAKE_INSTALL_CONFIG_NAME}"
                        COMMAND_ERROR_IS_FATAL ANY)
]=])
string(CONFIGURE "${_install_inner}" _install_inner @ONLY)
install(CODE "${_install_inner}")
