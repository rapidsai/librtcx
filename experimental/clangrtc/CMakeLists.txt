cmake_minimum_required(VERSION 3.25)
project(
    clangrtc
    VERSION 0.1.0
    LANGUAGES C CXX)

include(GNUInstallDirs)

option(BUILD_SHARED_LIBS "Build shared clangrtc and curtc libraries" ON)
option(CLANGRTC_STATIC_LINK_NVJITLINK "Link nvJitLink and its compiler backend statically" OFF)
option(CLANGRTC_STATIC_LINK_CUDART "Link the CUDA runtime statically in GPU tests" OFF)

set(CLANGRTC_CUDA_COMPILER_CACHE
    "${CMAKE_CURRENT_BINARY_DIR}/cuda-compiler-downloads"
    CACHE PATH "Pinned NVIDIA compiler library and download cache")

option(CLANGRTC_FETCH_LLVM "Build a private LLVM/Clang toolchain from source" ON)

if(CLANGRTC_FETCH_LLVM)
    include(cmake/BuildLLVM.cmake)
    return()
endif()

find_package(LLVM CONFIG REQUIRED)
find_package(Clang CONFIG REQUIRED)

option(CLANGRTC_BUILD_CURTC "Build the staged C CUDA JIT workflow" OFF)

include(cmake/Linkage.cmake)
if(LLVM_PACKAGE_VERSION VERSION_LESS 23)
    message(FATAL_ERROR "clangrtc requires LLVM/Clang 23 or newer")
endif()

set(CLANGRTC_RESOURCE_DIR
    "${LLVM_LIBRARY_DIR}/clang/${LLVM_VERSION_MAJOR}"
    CACHE PATH "Build-time Clang resource directory to embed")

if(NOT EXISTS "${CLANGRTC_RESOURCE_DIR}/include/stddef.h")
    set(CLANGRTC_RESOURCE_DIR
        "${LLVM_BINARY_DIR}/lib/clang/${LLVM_VERSION_MAJOR}"
        CACHE PATH "Build-time Clang resource directory to embed" FORCE)
endif()

if(NOT EXISTS "${CLANGRTC_RESOURCE_DIR}/include/stddef.h")
    message(FATAL_ERROR "Missing Clang resource headers to embed: ${CLANGRTC_RESOURCE_DIR}/include")
endif()

set(CLANGRTC_CUDA_PATH
    ""
    CACHE PATH "Default CUDA toolkit root (may be specified by API options)")

include(cmake/FetchxxHash.cmake)
include(cmake/FetchCUDACompilers.cmake)

clangrtc_fetch_nvvm()

include(cmake/EmbedHeaders.cmake)

add_library(clangrtc src/compiler.cpp "${CMAKE_CURRENT_BINARY_DIR}/clangrtc_headers.s")
add_dependencies(clangrtc clangrtc_headers)

# Preprocess assembly so compiler definitions and dependency flags are consumed normally.
set_source_files_properties("${CMAKE_CURRENT_BINARY_DIR}/clangrtc_headers.s"
                            PROPERTIES COMPILE_OPTIONS "-x;assembler-with-cpp")
                            
target_compile_definitions(
    clangrtc
    PRIVATE CLANGRTC_BUILDING_LIBRARY CLANGRTC_VERSION="${PROJECT_VERSION}"
            CLANGRTC_RESOURCE_DIR="${_clangrtc_resource_path}"
            CLANGRTC_CUDA_PATH="${CLANGRTC_CUDA_PATH}")
target_include_directories(
    clangrtc
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
           $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
target_include_directories(clangrtc SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS} ${CLANG_INCLUDE_DIRS})
target_compile_options(
    clangrtc PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-fno-rtti;-Wall;-Wextra;-Wno-unused-parameter>")
target_link_libraries(clangrtc PRIVATE ${_clangrtc_clang_libraries} ${_clangrtc_llvm_libraries}
                                       ${CMAKE_DL_LIBS})
target_include_directories(clangrtc SYSTEM PRIVATE "${CLANGRTC_NVVM_INCLUDE_DIR}")
target_compile_definitions(clangrtc
                           PRIVATE CLANGRTC_NVVM_LIBRARY="${CLANGRTC_NVVM_SELECTED_LIBRARY}")
add_library(clangrtc::clangrtc ALIAS clangrtc)

add_executable(clangrtc_cli src/cli.cpp)
set_target_properties(clangrtc_cli PROPERTIES OUTPUT_NAME clangrtc)
target_link_libraries(clangrtc_cli PRIVATE clangrtc)
target_compile_features(clangrtc_cli PRIVATE cxx_std_20)

# This optional component owns linking and caches; the low-level compiler stays independent.
if(CLANGRTC_BUILD_CURTC)
    clangrtc_fetch_nvjitlink()
    clangrtc_fetch_cuda_driver_headers()
    set(_curtc_identity "curtc-1")
    foreach(
        _file IN
        ITEMS "${CLANGRTC_NVVM_SELECTED_LIBRARY}"
              "${clangrtc_xxhash_SOURCE_DIR}/xxhash.h"
              "${CLANGRTC_NVJITLINK_SELECTED_LIBRARY}"
              "${CMAKE_CURRENT_SOURCE_DIR}/src/compiler.cpp"
              "${CMAKE_CURRENT_SOURCE_DIR}/src/error.hpp"
              "${CMAKE_CURRENT_SOURCE_DIR}/src/artifact_package.hpp"
              "${CMAKE_CURRENT_SOURCE_DIR}/src/content_hash.hpp"
              "${CMAKE_CURRENT_SOURCE_DIR}/src/function_ref.hpp"
              "${CMAKE_CURRENT_SOURCE_DIR}/src/curtc.cpp"
              "${CMAKE_CURRENT_SOURCE_DIR}/include/clangrtc.h"
              "${CMAKE_CURRENT_SOURCE_DIR}/include/curtc.h"
              ${_clangrtc_embedded_sources})
        set_property(
            DIRECTORY
            APPEND
            PROPERTY CMAKE_CONFIGURE_DEPENDS "${_file}")
        file(SHA256 "${_file}" _sha)
        string(APPEND _curtc_identity "${_sha}")
    endforeach()
    foreach(_library IN LISTS _clangrtc_compiler_files)
        file(SHA256 "${_library}" _sha)
        string(APPEND _curtc_identity "${_sha}")
    endforeach()
    if(CLANGRTC_STATIC_LINK_NVJITLINK)
        file(SHA256 "${CLANGRTC_NVPTXCOMPILER_STATIC_LIBRARY}" _sha)
        string(APPEND _curtc_identity "${_sha}")
    endif()
    string(
        SHA256
            _curtc_identity
            "${_curtc_identity};${LLVM_PACKAGE_VERSION};${CMAKE_CXX_COMPILER_ID};${CMAKE_CXX_COMPILER_VERSION};${CMAKE_BUILD_TYPE};${CMAKE_CXX_FLAGS}"
    )
    add_library(curtc src/curtc.cpp)
    add_library(clangrtc::curtc ALIAS curtc)
    target_include_directories(curtc SYSTEM PRIVATE "${CLANGRTC_CUDA_DRIVER_INCLUDE_DIR}")
    target_compile_definitions(
        curtc
        PRIVATE CLANGRTC_BUILDING_LIBRARY CURTC_BUILD_ID="${_curtc_identity}"
                CLANGRTC_STATIC_LINK_NVJITLINK=$<BOOL:${CLANGRTC_STATIC_LINK_NVJITLINK}>
                CLANGRTC_NVJITLINK_LIBRARY="${CLANGRTC_NVJITLINK_SELECTED_LIBRARY}")
    target_link_libraries(
        curtc
        PUBLIC clangrtc
        PRIVATE ${CMAKE_DL_LIBS})
    target_include_directories(curtc SYSTEM PRIVATE "${CLANGRTC_NVJITLINK_INCLUDE_DIR}")
    if(CLANGRTC_STATIC_LINK_NVJITLINK)
        target_link_libraries(curtc PRIVATE clangrtc::nvJitLink)
    endif()
endif()

# Shared library defaults also keep static archives suitable for embedding in shared libraries.
foreach(_target IN ITEMS clangrtc curtc)
    if(TARGET ${_target})
        target_compile_features(${_target} PRIVATE cxx_std_20)
        target_include_directories(${_target} SYSTEM PRIVATE "${clangrtc_xxhash_SOURCE_DIR}")
        set_target_properties(
            ${_target}
            PROPERTIES POSITION_INDEPENDENT_CODE ON
                       CXX_VISIBILITY_PRESET hidden
                       VISIBILITY_INLINES_HIDDEN YES
                       INSTALL_RPATH "$ORIGIN"
                       INSTALL_RPATH_USE_LINK_PATH YES)
    endif()
endforeach()

include(CTest)
if(BUILD_TESTING)
    find_package(GTest CONFIG REQUIRED)
    add_subdirectory(tests)
endif()
install(
    TARGETS clangrtc clangrtc_cli
    EXPORT clangrtcTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(FILES include/clangrtc.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(
    FILES cmake/LLVM-LICENSE.txt
    DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/clangrtc
    RENAME LLVM-LICENSE.txt)
if(CLANGRTC_BUILD_CURTC)
    install(
        TARGETS curtc
        EXPORT clangrtcTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    install(FILES include/curtc.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()
install(
    EXPORT clangrtcTargets
    NAMESPACE clangrtc::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/clangrtc)
include(CMakePackageConfigHelpers)
configure_package_config_file(
    cmake/clangrtcConfig.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/clangrtcConfig.cmake"
    PATH_VARS CMAKE_INSTALL_LIBDIR
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/clangrtc)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/clangrtcConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/clangrtcConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/clangrtcConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/clangrtc)
clangrtc_install_cuda_compilers()

option(CLANGRTC_BUILD_DOCS "Build the public C API reference with Doxygen" OFF)
if(CLANGRTC_BUILD_DOCS)
    find_package(Doxygen 1.9 REQUIRED)
    set(DOXYGEN_PROJECT_BRIEF "CUDA and host runtime compilation APIs")
    set(DOXYGEN_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/docs")
    set(DOXYGEN_OPTIMIZE_OUTPUT_FOR_C YES)
    set(DOXYGEN_GENERATE_XML YES)
    set(DOXYGEN_HAVE_DOT NO)
    set(DOXYGEN_QUIET YES)
    set(DOXYGEN_WARN_NO_PARAMDOC YES)
    set(DOXYGEN_WARN_AS_ERROR YES)
    set(_clangrtc_public_docs include/clangrtc.h)
    if(CLANGRTC_BUILD_CURTC)
        list(APPEND _clangrtc_public_docs include/curtc.h)
    endif()
    doxygen_add_docs(clangrtc_docs ${_clangrtc_public_docs}
                     COMMENT "Building clangrtc C API documentation (warnings are errors)")
endif()
