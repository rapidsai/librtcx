# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

include_guard(GLOBAL)

include(${CMAKE_CURRENT_LIST_DIR}/embed.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/compute_matrix_product.cmake)

function(add_jit_lto_kernel kernel_target)
  set(options)
  set(one_value KERNEL_FILE)
  set(multi_value LINK_LIBRARIES EXTRA_COMPILE_OPTIONS)

  cmake_parse_arguments(_JIT_LTO "${options}" "${one_value}" "${multi_value}" ${ARGN})

  add_library(${kernel_target} OBJECT EXCLUDE_FROM_ALL "${_JIT_LTO_KERNEL_FILE}")
  # Do not modify these properties, options, and libraries. Usage requirements (including CUDA
  # version, etc.) should be propagated to the kernel targets via INTERFACE libraries passed in
  # through the LINK_LIBRARIES argument.
  target_link_libraries(${kernel_target} PRIVATE ${_JIT_LTO_LINK_LIBRARIES})
  target_compile_options(${kernel_target} PRIVATE -Xfatbin=--compress-all --compress-mode=size)
  if(_JIT_LTO_EXTRA_COMPILE_OPTIONS)
    target_compile_options(${kernel_target} PRIVATE ${_JIT_LTO_EXTRA_COMPILE_OPTIONS})
  endif()
  set_target_properties(${kernel_target}
                        PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_FATBIN_COMPILATION ON
                                   POSITION_INDEPENDENT_CODE ON INTERPROCEDURAL_OPTIMIZATION ON)
endfunction()

#[=======================================================================[.rst:
rtcx_jit_lto_embed_begin
------------------------

Start a collection of JIT LTO kernel objects that will be embedded in one ELF
section. Calls to ``generate_jit_lto_kernels`` in the same CMake directory
register their fragments with this collection until it is finalized.
#]=======================================================================]
function(rtcx_jit_lto_embed_begin collection)
  set(options)
  set(one_value OUTPUT_DIRECTORY)
  cmake_parse_arguments(ARG "${options}" "${one_value}" "" ${ARGN})

  if(NOT ARG_OUTPUT_DIRECTORY)
    message(FATAL_ERROR "OUTPUT_DIRECTORY is required for rtcx_jit_lto_embed_begin()")
  endif()
  if(NOT collection MATCHES "^[A-Za-z_][A-Za-z0-9_]*$")
    message(FATAL_ERROR "JIT LTO embed collection '${collection}' is not a valid C++ identifier")
  endif()

  get_property(active_collection DIRECTORY PROPERTY RTCX_ACTIVE_JIT_LTO_EMBED_COLLECTION)
  if(NOT "${active_collection}" STREQUAL "")
    message(FATAL_ERROR "JIT LTO embed collection '${active_collection}' is already active")
  endif()

  rtcx_add_embed(${collection})
  set_property(TARGET ${collection}__embed_props PROPERTY RTCX_JIT_LTO_OUTPUT_DIRECTORY
                                                             "${ARG_OUTPUT_DIRECTORY}")
  set_property(DIRECTORY PROPERTY RTCX_ACTIVE_JIT_LTO_EMBED_COLLECTION "${collection}")
endfunction()

#[=======================================================================[.rst:
rtcx_jit_lto_embed_finalize
---------------------------

Generate the consolidated ELF blob and the sole translation unit defining the
static fragment-entry members for a JIT LTO embedding collection.
#]=======================================================================]
function(rtcx_jit_lto_embed_finalize collection)
  set(options)
  set(one_value SOURCE_LIST)
  cmake_parse_arguments(ARG "${options}" "${one_value}" "" ${ARGN})

  if(NOT ARG_SOURCE_LIST)
    message(FATAL_ERROR "SOURCE_LIST is required for rtcx_jit_lto_embed_finalize()")
  endif()

  get_property(active_collection DIRECTORY PROPERTY RTCX_ACTIVE_JIT_LTO_EMBED_COLLECTION)
  if(NOT "${active_collection}" STREQUAL "${collection}")
    message(FATAL_ERROR "JIT LTO embed collection '${collection}' is not active")
  endif()

  get_property(output_directory TARGET ${collection}__embed_props
               PROPERTY RTCX_JIT_LTO_OUTPUT_DIRECTORY)
  get_property(fragment_tags TARGET ${collection}__embed_props PROPERTY RTCX_JIT_LTO_FRAGMENT_TAGS)
  get_property(fragment_headers TARGET ${collection}__embed_props
               PROPERTY RTCX_JIT_LTO_FRAGMENT_TAG_HEADERS)

  if("${fragment_tags}" STREQUAL "")
    message(FATAL_ERROR "JIT LTO embed collection '${collection}' contains no fragments")
  endif()

  # The LTO objects must remain uncompressed because nvJitLink consumes them directly.
  rtcx_embed(${collection} COMPRESSION none OUTPUT_DIRECTORY "${output_directory}")

  set(fragment_tag_header_files "")
  foreach(header_file IN LISTS fragment_headers)
    if(NOT header_file MATCHES "^(\".*\"|<.*>)$")
      set(header_file "\"${header_file}\"")
    endif()
    list(APPEND fragment_tag_header_files "${header_file}")
  endforeach()
  list(REMOVE_DUPLICATES fragment_tag_header_files)

  set(fragment_tag_includes "")
  foreach(header_file IN LISTS fragment_tag_header_files)
    string(APPEND fragment_tag_includes "#include ${header_file}\n")
  endforeach()

  set(fragment_definitions "")
  set(fragment_index 0)
  foreach(fragment_tag IN LISTS fragment_tags)
    string(APPEND fragment_definitions
           "namespace {\nusing fragment_entry_${fragment_index} = rtcx::static_fatbin_fragment_entry<${fragment_tag}>;\n}  // namespace\n\n"
           "template <>\nconst uint8_t* const fragment_entry_${fragment_index}::data = ${collection}::files.data() + ${collection}::file_ranges[${fragment_index}][0];\n\n"
           "template <>\nconst size_t fragment_entry_${fragment_index}::length = ${collection}::file_ranges[${fragment_index}][1];\n\n")
    math(EXPR fragment_index "${fragment_index} + 1")
  endforeach()

  set(embed_header_file "${output_directory}/${collection}.hpp")
  set(registration_source "${output_directory}/${collection}_fragments.cpp")
  configure_file("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/register_embedded_fatbins.cpp.in"
                 "${registration_source}" @ONLY)

  # The registration source includes a header produced by rtcx_embed(). Record the dependency
  # explicitly so a parallel build cannot compile it before the blob exists.
  set_source_files_properties(
    "${registration_source}"
    PROPERTIES OBJECT_DEPENDS
               "${embed_header_file};${output_directory}/${collection}.s;${output_directory}/${collection}.bin")

  set(${ARG_SOURCE_LIST}
      "${${ARG_SOURCE_LIST}}" "${registration_source}" "${output_directory}/${collection}.s"
      PARENT_SCOPE)
  set_property(DIRECTORY PROPERTY RTCX_ACTIVE_JIT_LTO_EMBED_COLLECTION "")
endfunction()

function(process_jit_lto_matrix_entry source_list_var)
  set(options)
  set(one_value NAME_FORMAT KERNEL_INPUT_FILE OUTPUT_DIRECTORY FRAGMENT_TAG_FORMAT
                MATRIX_JSON_ENTRY)
  set(multi_value KERNEL_LINK_LIBRARIES FRAGMENT_TAG_HEADER_FILES KERNEL_EXTRA_COMPILE_OPTIONS)

  cmake_parse_arguments(_JIT_LTO "${options}" "${one_value}" "${multi_value}" ${ARGN})

  populate_matrix_variables("${_JIT_LTO_MATRIX_JSON_ENTRY}")
  string(CONFIGURE "${_JIT_LTO_NAME_FORMAT}" kernel_name @ONLY)
  string(CONFIGURE "${_JIT_LTO_FRAGMENT_TAG_FORMAT}" fragment_tag @ONLY)

  set(kernel_file "${_JIT_LTO_OUTPUT_DIRECTORY}/${kernel_name}_kernel.cu")
  set(kernel_target "${kernel_name}_kernel")
  configure_file("${_JIT_LTO_KERNEL_INPUT_FILE}" "${kernel_file}" @ONLY)

  add_jit_lto_kernel(${kernel_target} KERNEL_FILE "${kernel_file}"
                     LINK_LIBRARIES ${_JIT_LTO_KERNEL_LINK_LIBRARIES}
                     EXTRA_COMPILE_OPTIONS ${_JIT_LTO_KERNEL_EXTRA_COMPILE_OPTIONS})

  get_property(collection DIRECTORY PROPERTY RTCX_ACTIVE_JIT_LTO_EMBED_COLLECTION)
  if("${collection}" STREQUAL "")
    message(FATAL_ERROR
            "generate_jit_lto_kernels() requires an active rtcx_jit_lto_embed_begin() collection")
  endif()

  rtcx_embed_blob(${collection} ID "${kernel_name}" FILE "$<TARGET_OBJECTS:${kernel_target}>"
                  DEST "${kernel_name}")
  set_property(TARGET ${collection}__embed_props APPEND PROPERTY RTCX_JIT_LTO_FRAGMENT_TAGS
                                                             "${fragment_tag}")
  set_property(TARGET ${collection}__embed_props APPEND PROPERTY RTCX_JIT_LTO_FRAGMENT_TAG_HEADERS
                                                             ${_JIT_LTO_FRAGMENT_TAG_HEADER_FILES})
  set(${source_list_var} "${${source_list_var}}" PARENT_SCOPE)
endfunction()

function(generate_jit_lto_kernels source_list_var)
  set(options)
  set(one_value NAME_FORMAT MATRIX_JSON_FILE MATRIX_JSON_STRING KERNEL_INPUT_FILE
                FRAGMENT_TAG_FORMAT OUTPUT_DIRECTORY)
  set(multi_value KERNEL_LINK_LIBRARIES FRAGMENT_TAG_HEADER_FILES KERNEL_EXTRA_COMPILE_OPTIONS)

  cmake_parse_arguments(_JIT_LTO "${options}" "${one_value}" "${multi_value}" ${ARGN})

  if(_JIT_LTO_MATRIX_JSON_FILE)
    set_property(DIRECTORY PROPERTY CMAKE_CONFIGURE_DEPENDS "${_JIT_LTO_MATRIX_JSON_FILE}" APPEND)
    compute_matrix_product(matrix_product MATRIX_JSON_FILE "${_JIT_LTO_MATRIX_JSON_FILE}")
  else()
    compute_matrix_product(matrix_product MATRIX_JSON_STRING "${_JIT_LTO_MATRIX_JSON_STRING}")
  endif()

  string(JSON len LENGTH "${matrix_product}")
  math(EXPR last "${len} - 1")

  # cmake-lint: disable=C0103,E1120
  foreach(i RANGE "${last}")
    string(JSON matrix_json_entry GET "${matrix_product}" "${i}")
    process_jit_lto_matrix_entry("${source_list_var}"
                                 NAME_FORMAT
                                 "${_JIT_LTO_NAME_FORMAT}"
                                 KERNEL_INPUT_FILE
                                 "${_JIT_LTO_KERNEL_INPUT_FILE}"
                                 FRAGMENT_TAG_FORMAT
                                 "${_JIT_LTO_FRAGMENT_TAG_FORMAT}"
                                 FRAGMENT_TAG_HEADER_FILES
                                 ${_JIT_LTO_FRAGMENT_TAG_HEADER_FILES}
                                 OUTPUT_DIRECTORY
                                 "${_JIT_LTO_OUTPUT_DIRECTORY}"
                                 MATRIX_JSON_ENTRY
                                 "${matrix_json_entry}"
                                 KERNEL_LINK_LIBRARIES
                                 ${_JIT_LTO_KERNEL_LINK_LIBRARIES}
                                 KERNEL_EXTRA_COMPILE_OPTIONS
                                 ${_JIT_LTO_KERNEL_EXTRA_COMPILE_OPTIONS})
  endforeach()

  set(${source_list_var} "${${source_list_var}}" PARENT_SCOPE)
endfunction()
