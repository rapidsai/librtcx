# SPDX-License-Identifier: Apache-2.0

# Build LLVM in its own project so its build targets do not enter this project.
include(ExternalProject)
set(CLANGRTC_LLVM_VERSION
    "23.1.5"
    CACHE STRING "Pinned stable LLVM source version")
set(CLANGRTC_LLVM_SHA256
    "fb50b1eac93ddeb3aa8dab4cfe324dff74f4a9c2bdee6160c9e36188f8d06a6e"
    CACHE STRING "Published SHA-256 of the selected LLVM source archive")
set(CLANGRTC_LLVM_SOURCE_DIR
    ""
    CACHE PATH "Existing llvm-project source tree; otherwise download the pinned release")
set(CLANGRTC_LLVM_CACHE
    "${CMAKE_CURRENT_BINARY_DIR}/llvm-downloads"
    CACHE PATH "Private LLVM build and download cache")
set(CLANGRTC_LLVM_BUILD_JOBS
    6
    CACHE STRING "Maximum parallel jobs for the private LLVM build")

if(CLANGRTC_LLVM_SOURCE_DIR)
    if(NOT EXISTS "${CLANGRTC_LLVM_SOURCE_DIR}/llvm/CMakeLists.txt")
        message(FATAL_ERROR "CLANGRTC_LLVM_SOURCE_DIR must name an llvm-project source tree")
    endif()
    include("${CLANGRTC_LLVM_SOURCE_DIR}/cmake/Modules/LLVMVersion.cmake")
    set(_source_version "${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}.${LLVM_VERSION_PATCH}")
    if(NOT _source_version VERSION_EQUAL CLANGRTC_LLVM_VERSION)
        message(
            FATAL_ERROR
                "LLVM source version ${_source_version} does not match pinned version ${CLANGRTC_LLVM_VERSION}"
        )
    endif()
    set(CLANGRTC_LLVM_FETCHED_SOURCE "${CLANGRTC_LLVM_SOURCE_DIR}")
else()
    set(_archive "${CLANGRTC_LLVM_CACHE}/llvm-project-${CLANGRTC_LLVM_VERSION}.src.tar.xz")
    file(MAKE_DIRECTORY "${CLANGRTC_LLVM_CACHE}")
    if(EXISTS "${_archive}")
        file(SHA256 "${_archive}" _checksum)
        if(NOT _checksum STREQUAL CLANGRTC_LLVM_SHA256)
            message(FATAL_ERROR "Cached LLVM archive checksum mismatch: ${_archive}")
        endif()
    else()
        file(
            DOWNLOAD
            "https://github.com/llvm/llvm-project/releases/download/llvmorg-${CLANGRTC_LLVM_VERSION}/llvm-project-${CLANGRTC_LLVM_VERSION}.src.tar.xz"
            "${_archive}"
            EXPECTED_HASH "SHA256=${CLANGRTC_LLVM_SHA256}"
            TLS_VERIFY ON
            STATUS _status)
        list(GET _status 0 _code)
        if(NOT _code EQUAL 0)
            message(FATAL_ERROR "LLVM download failed: ${_status}")
        endif()
    endif()
    set(CLANGRTC_LLVM_FETCHED_SOURCE
        "${CLANGRTC_LLVM_CACHE}/llvm-project-${CLANGRTC_LLVM_VERSION}.src")
    if(NOT EXISTS "${CLANGRTC_LLVM_FETCHED_SOURCE}/llvm/CMakeLists.txt")
        file(ARCHIVE_EXTRACT INPUT "${_archive}" DESTINATION "${CLANGRTC_LLVM_CACHE}")
    endif()
endif()

set(_llvm_install "${CLANGRTC_LLVM_CACHE}/install-${CLANGRTC_LLVM_VERSION}")
string(REGEX MATCH "^[0-9]+" _llvm_major "${CLANGRTC_LLVM_VERSION}")
# Preserve compiler and system-dependency choices in the dependency build too.
get_cmake_property(_cache_variables CACHE_VARIABLES)
set(_llvm_args)
foreach(_name IN LISTS _cache_variables)
    get_property(
        _type
        CACHE "${_name}"
        PROPERTY TYPE)
    if(NOT _type STREQUAL "INTERNAL"
       AND _name
           MATCHES
           "^(CMAKE_(C_|CXX_|EXE_LINKER_|SHARED_LINKER_|TOOLCHAIN_FILE|PREFIX_PATH)|LLVM_ENABLE_(ZLIB|ZSTD|LIBXML2)$|LLVM_USE_STATIC_ZSTD$|ZLIB_|zstd_|LibXml2_)"
    )
        string(REPLACE ";" "$<SEMICOLON>" _value "${${_name}}")
        list(APPEND _llvm_args "-D${_name}:${_type}=${_value}")
    endif()
endforeach()

ExternalProject_Add(
    clangrtc_llvm
    SOURCE_DIR "${CLANGRTC_LLVM_FETCHED_SOURCE}/llvm"
    BINARY_DIR "${CLANGRTC_LLVM_CACHE}/build-${CLANGRTC_LLVM_VERSION}"
    CMAKE_ARGS ${_llvm_args}
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
               "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
               "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
               "-DCMAKE_INSTALL_PREFIX=${_llvm_install}"
    BUILD_COMMAND "${CMAKE_COMMAND}" --build <BINARY_DIR> --parallel "${CLANGRTC_LLVM_BUILD_JOBS}"
    INSTALL_COMMAND "${CMAKE_COMMAND}" --install <BINARY_DIR>)

# Forward caller settings, including dependency linkage and toolchain paths, to the actual library
# build. Compiler package paths always come from our build.
get_cmake_property(_cache_variables CACHE_VARIABLES)
set(_inner_args)
foreach(_name IN LISTS _cache_variables)
    get_property(
        _type
        CACHE "${_name}"
        PROPERTY TYPE)
    if(_type STREQUAL "INTERNAL"
       OR _name
          MATCHES
          "^(LLVM_DIR|Clang_DIR|MLIR_DIR|CLANGRTC_FETCH_LLVM|CLANGRTC_RESOURCE_DIR|CLANGRTC_LLVM_.*)$"
    )
        continue()
    endif()
    if(_name
       MATCHES
       "^(CLANGRTC_|CURTC_|NVVM_|NVJitLink_|xxHash_|CUDAToolkit_|FETCHCONTENT_|CMAKE_(BUILD_TYPE|C_|CXX_|EXE_LINKER_|SHARED_LINKER_|TOOLCHAIN_FILE|INSTALL_|PREFIX_PATH)|BUILD_|GTest_DIR|ZLIB_USE_STATIC_LIBS)"
    )
        string(REPLACE ";" "$<SEMICOLON>" _value "${${_name}}")
        list(APPEND _inner_args "-D${_name}:${_type}=${_value}")
    endif()
endforeach()

set(_inner_build "${CMAKE_CURRENT_BINARY_DIR}/clangrtc-build")
ExternalProject_Add(
    clangrtc_libraries
    DEPENDS clangrtc_llvm
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
    BINARY_DIR "${_inner_build}"
    CMAKE_ARGS ${_inner_args} -DCLANGRTC_FETCH_LLVM=OFF "-DLLVM_DIR=${_llvm_install}/lib/cmake/llvm"
               "-DClang_DIR=${_llvm_install}/lib/cmake/clang"
               "-DCLANGRTC_RESOURCE_DIR=${_llvm_install}/lib/clang/${_llvm_major}"
    BUILD_COMMAND "${CMAKE_COMMAND}" --build <BINARY_DIR> --parallel "${CLANGRTC_LLVM_BUILD_JOBS}"
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
