# SPDX-License-Identifier: Apache-2.0

# @file tests/clangrtc/compiler_linkage/CMakeLists.txt
#
# @brief Construct synthetic LLVM/Clang packages for static-linkage policy tests.
#
# CASE selects valid archives, direct or transitive shared dependencies, wrappers, mislabeled files,
# or a missing archive; no compiler execution is needed.

cmake_minimum_required(VERSION 3.25)
project(CompilerLinkage LANGUAGES NONE)

# No compiler is required: configuration alone must reject incompatible packages.
set(CMAKE_STATIC_LIBRARY_SUFFIX .a)
function(llvm_map_components_to_libnames output)
  set(${output} LLVMCore PARENT_SCOPE)
endfunction()

foreach(target IN ITEMS LLVMCore clangCodeGen clangFrontend clangDriver clangSerialization
                        clangSema)
  set(type STATIC)
  if((CASE STREQUAL "shared_llvm" AND target STREQUAL "LLVMCore")
     OR (CASE STREQUAL "shared_clang" AND target STREQUAL "clangCodeGen"))
    set(type SHARED)
  endif()
  add_library(${target} ${type} IMPORTED)
  set(path "${CMAKE_CURRENT_BINARY_DIR}/lib${target}.a")
  file(WRITE "${path}" "")
  set_target_properties(${target} PROPERTIES IMPORTED_LOCATION "${path}")
endforeach()

if(CASE STREQUAL "transitive_shared" OR CASE STREQUAL "wrapped_shared")
  add_library(LLVMBackend SHARED IMPORTED)
  if(CASE STREQUAL "wrapped_shared")
    add_library(compiler_wrapper INTERFACE IMPORTED)
    set_target_properties(compiler_wrapper PROPERTIES INTERFACE_LINK_LIBRARIES LLVMBackend)
    set(dependency "$<LINK_ONLY:$<TARGET_NAME_IF_EXISTS:compiler_wrapper>>")
  else()
    set(dependency LLVMBackend)
  endif()
  set_target_properties(LLVMCore PROPERTIES INTERFACE_LINK_LIBRARIES "${dependency}")
elseif(CASE STREQUAL "raw_shared" OR CASE STREQUAL "disguised_static")
  set(path "${CMAKE_CURRENT_BINARY_DIR}/libLLVM.so")
  file(WRITE "${path}" "")
  if(CASE STREQUAL "raw_shared")
    set_target_properties(LLVMCore PROPERTIES INTERFACE_LINK_LIBRARIES "${path}")
  else()
    set_target_properties(LLVMCore PROPERTIES IMPORTED_LOCATION "${path}")
  endif()
elseif(CASE STREQUAL "missing_archive")
  set(path "${CMAKE_CURRENT_BINARY_DIR}/missing.a")
  set_target_properties(LLVMCore PROPERTIES IMPORTED_LOCATION "${path}")
endif()

include("${MODULE}")
