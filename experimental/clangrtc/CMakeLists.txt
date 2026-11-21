cmake_minimum_required(VERSION 3.25)
project(clangrtc VERSION 0.1.0 LANGUAGES C CXX)
include(GNUInstallDirs)
option(BUILD_SHARED_LIBS "Build shared clangrtc and curtc libraries" ON)
option(CLANGRTC_STATIC_LINK_NVVM "Require a static libNVVM archive (not distributed by NVIDIA)" OFF)
option(CLANGRTC_STATIC_LINK_NVJITLINK "Link nvJitLink and its compiler backend statically" OFF)
option(CLANGRTC_STATIC_LINK_CUDART "Link the CUDA runtime statically in GPU tests" OFF)
option(CLANGRTC_STATIC_LINK_XXHASH
       "Require Cyan xxHash as a static archive with x86 runtime dispatch" OFF)
option(CLANGRTC_FETCH_CUDA_COMPILERS
       "Fetch pinned NVIDIA compiler libraries unless explicitly supplied" ON)
set(CLANGRTC_CUDA_COMPILER_CACHE "${CMAKE_CURRENT_BINARY_DIR}/cuda-compiler-downloads"
    CACHE PATH "Pinned NVIDIA compiler library and download cache")

option(CLANGRTC_FETCH_LLVM "Build a private LLVM/Clang toolchain from source" ON)
if(CLANGRTC_FETCH_LLVM)
  include(cmake/BuildLLVM.cmake)
  return()
endif()
find_package(LLVM CONFIG REQUIRED)
# Accommodate an existing Clang build that enables CIR; clangrtc does not use it.
find_package(MLIR CONFIG QUIET HINTS "${LLVM_DIR}/../mlir")
find_package(Clang CONFIG REQUIRED)
option(CLANGRTC_BUILD_CURTC "Build the staged C CUDA JIT workflow" OFF)
include(cmake/Linkage.cmake)
if(LLVM_PACKAGE_VERSION VERSION_LESS 23)
  message(FATAL_ERROR "clangrtc requires LLVM/Clang 23 or newer")
endif()
set(CLANGRTC_RESOURCE_DIR "${LLVM_LIBRARY_DIR}/clang/${LLVM_VERSION_MAJOR}"
    CACHE PATH "Clang resource directory")
if(NOT EXISTS "${CLANGRTC_RESOURCE_DIR}/include/stddef.h")
  set(CLANGRTC_RESOURCE_DIR "${LLVM_BINARY_DIR}/lib/clang/${LLVM_VERSION_MAJOR}"
      CACHE PATH "Clang resource directory" FORCE)
endif()
set(CLANGRTC_CUDA_PATH "" CACHE PATH "Default CUDA toolkit root (may be specified by API options)")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(xxHash REQUIRED)
find_package(NVVM REQUIRED)
if(CMAKE_CROSSCOMPILING AND NOT CMAKE_CROSSCOMPILING_EMULATOR)
  message(FATAL_ERROR "NVVM compatibility validation requires native execution or CMAKE_CROSSCOMPILING_EMULATOR"
  )
endif()
# Do not cache probe results across changes of the LLVM producer or NVVM reader.
unset(_nvvm_runs CACHE)
unset(_nvvm_compiles CACHE)
try_run(_nvvm_runs _nvvm_compiles "${CMAKE_CURRENT_BINARY_DIR}/nvvm-probe"
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/nvvm_probe.cpp"
        CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${LLVM_INCLUDE_DIRS}" LINK_LIBRARIES
                    ${_clangrtc_llvm_libraries} NVVM::NVVM
        COMPILE_OUTPUT_VARIABLE _nvvm_build_log
        RUN_OUTPUT_VARIABLE _nvvm_run_log)
if(NOT _nvvm_compiles OR NOT "${_nvvm_runs}" STREQUAL "0")
  message(FATAL_ERROR "libNVVM ${CLANGRTC_NVVM_SELECTED_LIBRARY} (headers ${CLANGRTC_NVVM_INCLUDE_DIR}) cannot consume LLVM ${LLVM_PACKAGE_VERSION} bitcode with -nvvm-latest -gen-lto. Select a compatible installation with NVVM_ROOT.\n${_nvvm_build_log}\n${_nvvm_run_log}"
  )
endif()
file(GLOB_RECURSE _headers CONFIGURE_DEPENDS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/headers"
     "headers/*")
set(_embedded
    "#pragma once\n#include <map>\n#include <string>\ninline std::map<std::string,std::string> builtin_headers(){return {\n"
)
foreach(_name IN LISTS _headers)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
                                         "${CMAKE_CURRENT_SOURCE_DIR}/headers/${_name}")
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/headers/${_name}" _contents)
  string(APPEND _embedded "{\"/__clangrtc__/include/${_name}\",R\"CRTC(${_contents})CRTC\"},\n")
endforeach()
string(APPEND _embedded "};}\n")
include(cmake/CUDAHeaderModes.cmake)
string(APPEND _embedded "${_cuda_modes}")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/embedded.hpp" "${_embedded}")
add_library(clangrtc src/compiler.cpp)
target_compile_features(clangrtc PRIVATE cxx_std_20)
target_compile_definitions(clangrtc
                           PRIVATE CLANGRTC_BUILDING_LIBRARY CLANGRTC_VERSION="${PROJECT_VERSION}"
                                   CLANGRTC_RESOURCE_DIR="${CLANGRTC_RESOURCE_DIR}"
                                   CLANGRTC_CUDA_PATH="${CLANGRTC_CUDA_PATH}")
target_include_directories(clangrtc PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                                           $<INSTALL_INTERFACE:include>
                           PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
target_include_directories(clangrtc SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS} ${CLANG_INCLUDE_DIRS})
target_compile_options(clangrtc PRIVATE -fno-rtti -Wall -Wextra -Wno-unused-parameter)
target_link_libraries(clangrtc PRIVATE ${_clangrtc_clang_libraries} ${_clangrtc_llvm_libraries}
                                       xxHash::xxhash ${CMAKE_DL_LIBS})
target_include_directories(clangrtc SYSTEM PRIVATE "${CLANGRTC_NVVM_INCLUDE_DIR}")
target_compile_definitions(clangrtc
                           PRIVATE CLANGRTC_STATIC_LINK_NVVM=$<BOOL:${CLANGRTC_STATIC_LINK_NVVM}>
                                   CLANGRTC_NVVM_LIBRARY="${CLANGRTC_NVVM_SELECTED_LIBRARY}")
if(CLANGRTC_STATIC_LINK_NVVM)
  target_link_libraries(clangrtc PRIVATE NVVM::NVVM)
endif()
set_target_properties(clangrtc PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(clangrtc PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES
                                          INSTALL_RPATH "$ORIGIN" INSTALL_RPATH_USE_LINK_PATH YES)
add_library(clangrtc::clangrtc ALIAS clangrtc)
add_executable(clangrtc_cli src/cli.cpp)
set_target_properties(clangrtc_cli PROPERTIES OUTPUT_NAME clangrtc)
target_link_libraries(clangrtc_cli PRIVATE clangrtc)
target_compile_features(clangrtc_cli PRIVATE cxx_std_20)
# This optional component owns linking and caches; the low-level compiler stays independent.
if(CLANGRTC_BUILD_CURTC)
  find_package(NVJitLink REQUIRED)
  if(CLANGRTC_FETCH_CUDA_COMPILERS)
    include(cmake/FetchCUDACompilers.cmake)
    clangrtc_fetch_cuda_driver_headers()
  endif()
  find_path(CLANGRTC_CUDA_DRIVER_INCLUDE_DIR cuda.h HINTS ${CUDAToolkit_INCLUDE_DIRS}
                                                          "${CLANGRTC_CUDA_PATH}/include"
            DOC "Private CUDA 12.4+ driver headers for curtc")
  if(NOT CLANGRTC_CUDA_DRIVER_INCLUDE_DIR)
    message(FATAL_ERROR "curtc requires CUDA 12.4+ driver headers; enable CLANGRTC_FETCH_CUDA_COMPILERS or set CLANGRTC_CUDA_DRIVER_INCLUDE_DIR"
    )
  endif()
  file(STRINGS "${CLANGRTC_CUDA_DRIVER_INCLUDE_DIR}/cuda.h" _driver_version_line
       REGEX "^#define CUDA_VERSION +[0-9]+")
  if(NOT _driver_version_line MATCHES "CUDA_VERSION +([0-9]+)" OR CMAKE_MATCH_1 LESS 12040)
    message(FATAL_ERROR "curtc requires CUDA 12.4+ driver headers in CLANGRTC_CUDA_DRIVER_INCLUDE_DIR"
    )
  endif()
  set(_curtc_identity "curtc-1")
  foreach(_file IN
          ITEMS "${CLANGRTC_NVVM_SELECTED_LIBRARY}"
                "${CLANGRTC_XXHASH_SELECTED_LIBRARY}"
                "${CLANGRTC_NVJITLINK_SELECTED_LIBRARY}"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/compiler.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/error.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/artifact_package.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/content_hash.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/function_ref.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/curtc.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/clangrtc.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/curtc.h"
                "${CMAKE_CURRENT_BINARY_DIR}/embedded.hpp")
    if(NOT _file STREQUAL "${CMAKE_CURRENT_BINARY_DIR}/embedded.hpp")
      set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_file}")
    endif()
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
  string(SHA256 _curtc_identity
                "${_curtc_identity};${LLVM_PACKAGE_VERSION};${CMAKE_CXX_COMPILER_ID};${CMAKE_CXX_COMPILER_VERSION};${CMAKE_BUILD_TYPE};${CMAKE_CXX_FLAGS}"
  )
  add_library(curtc src/curtc.cpp)
  add_library(clangrtc::curtc ALIAS curtc)
  set_target_properties(curtc
                        PROPERTIES EXPORT_NAME curtc
                                   INSTALL_RPATH_USE_LINK_PATH YES
                                   INSTALL_RPATH "$ORIGIN"
                                   CXX_VISIBILITY_PRESET hidden
                                   VISIBILITY_INLINES_HIDDEN YES)
  target_compile_features(curtc PRIVATE cxx_std_20)
  target_include_directories(curtc SYSTEM PRIVATE "${CLANGRTC_CUDA_DRIVER_INCLUDE_DIR}")
  target_compile_definitions(curtc
                             PRIVATE CLANGRTC_BUILDING_LIBRARY
                                     CURTC_BUILD_ID="${_curtc_identity}"
                                     CLANGRTC_STATIC_LINK_NVJITLINK=$<BOOL:${CLANGRTC_STATIC_LINK_NVJITLINK}>
                                     CLANGRTC_NVJITLINK_LIBRARY="${CLANGRTC_NVJITLINK_SELECTED_LIBRARY}"
  )
  target_include_directories(curtc SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})
  target_link_libraries(curtc PUBLIC clangrtc PRIVATE ${_clangrtc_llvm_libraries} xxHash::xxhash
                                                      ${CMAKE_DL_LIBS})
  target_include_directories(curtc SYSTEM PRIVATE "${CLANGRTC_NVJITLINK_INCLUDE_DIR}")
  if(CLANGRTC_STATIC_LINK_NVJITLINK)
    target_link_libraries(curtc PRIVATE NVJitLink::NVJitLink)
  endif()
  set_target_properties(curtc PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
include(CTest)
if(BUILD_TESTING)
  find_package(GTest CONFIG REQUIRED)
  add_subdirectory(tests)
endif()
install(TARGETS clangrtc clangrtc_cli
        EXPORT clangrtcTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(FILES include/clangrtc.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
if(CLANGRTC_BUILD_CURTC)
  install(TARGETS curtc
          EXPORT clangrtcTargets
          ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
          LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
          RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  install(FILES include/curtc.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()
install(EXPORT clangrtcTargets NAMESPACE clangrtc::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/clangrtc)
include(CMakePackageConfigHelpers)
configure_package_config_file(cmake/clangrtcConfig.cmake.in
                              "${CMAKE_CURRENT_BINARY_DIR}/clangrtcConfig.cmake"
                              PATH_VARS CMAKE_INSTALL_LIBDIR
                              INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/clangrtc)
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/clangrtcConfigVersion.cmake"
                                 VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/clangrtcConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/clangrtcConfigVersion.cmake"
              cmake/FindNVVM.cmake
              cmake/FindNVJitLink.cmake
              cmake/FindxxHash.cmake
              cmake/FetchCUDACompilers.cmake
              cmake/cuda-compiler-libraries.json
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/clangrtc)
if(CLANGRTC_FETCH_CUDA_COMPILERS)
  include(cmake/FetchCUDACompilers.cmake)
  clangrtc_install_cuda_compilers()
endif()

option(CLANGRTC_BUILD_DOCS "Build the public C API reference with Doxygen" OFF)
if(CLANGRTC_BUILD_DOCS)
  find_package(Doxygen 1.9 REQUIRED)
  set(DOXYGEN_PROJECT_BRIEF "CUDA and host runtime compilation APIs")
  set(DOXYGEN_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/docs")
  set(DOXYGEN_OPTIMIZE_OUTPUT_FOR_C YES)
  set(DOXYGEN_EXTRACT_ALL NO)
  set(DOXYGEN_GENERATE_HTML YES)
  set(DOXYGEN_GENERATE_LATEX NO)
  set(DOXYGEN_GENERATE_XML YES)
  set(DOXYGEN_HAVE_DOT NO)
  set(DOXYGEN_QUIET YES)
  set(DOXYGEN_WARN_IF_UNDOCUMENTED YES)
  set(DOXYGEN_WARN_IF_DOC_ERROR YES)
  set(DOXYGEN_WARN_NO_PARAMDOC YES)
  set(DOXYGEN_WARN_AS_ERROR YES)
  set(_clangrtc_public_docs include/clangrtc.h)
  if(CLANGRTC_BUILD_CURTC)
    list(APPEND _clangrtc_public_docs include/curtc.h)
  endif()
  doxygen_add_docs(clangrtc_docs ${_clangrtc_public_docs}
                   COMMENT "Building clangrtc C API documentation (warnings are errors)")
endif()
