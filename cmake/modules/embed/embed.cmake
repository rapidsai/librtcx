# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include_guard(GLOBAL)

#[=======================================================================[.rst:
rtcx_embed
----------

Finalize an embed target, generating build artifacts for JIT embedding.

.. code-block:: cmake

  rtcx_embed(<target>
      COMPRESSION <none|zstd>
      OUTPUT_DIRECTORY <dir>)

Finalizes the embed target ``<target>`` by configuring a runner executable and
custom build step that produces the embedded data artifacts. Must be called
after ``rtcx_add_embed`` and any desired calls to ``rtcx_embed_includes`` or
``rtcx_embed_blob``.

.. note::

  The ``zstd`` and ``xxhash`` CMake targets must be available before calling
  ``rtcx_embed``.

``<target>``
  Required. Name of the embed target, previously initialized with ``rtcx_add_embed``.

``COMPRESSION``
  Required. Compression method for the embedded data. Must be ``none`` or
  ``zstd``.

``OUTPUT_DIRECTORY``
  Required. Directory where the generated artifacts will be written.

Result Variables
^^^^^^^^^^^^^^^^

``<target>_INCLUDE_DIRS``
  Set in the calling scope to ``OUTPUT_DIRECTORY``. Add to target include
  directories to access the generated header.

``<target>_SOURCE_DIR``
  Set in the calling scope to ``OUTPUT_DIRECTORY``.

Generated Files
^^^^^^^^^^^^^^^

The following files are created in ``OUTPUT_DIRECTORY``:

``<target>.hpp``
  C++ header declaring the embedded data symbols.

``<target>.s``
  assembly file containing the embedded data.

``<target>.bin``
  raw binary of the embedded data.
#]=======================================================================]
# cmake-lint: disable=R0915
function(rtcx_embed TARGET)
  set(OPTIONS "")
  set(ONE_VALUE_ARGS "COMPRESSION" "OUTPUT_DIRECTORY")
  set(MULTI_VALUE_ARGS "")
  cmake_parse_arguments(ARG "${OPTIONS}" "${ONE_VALUE_ARGS}" "${MULTI_VALUE_ARGS}" ${ARGN})

  if(NOT TARGET zstd)
    message(FATAL_ERROR "zstd target is required for rtcx_embed().")
  endif()
  if(NOT TARGET xxhash)
    message(FATAL_ERROR "xxhash target is required for rtcx_embed().")
  endif()

  if(NOT TARGET ${TARGET}__embed_props)
    message(FATAL_ERROR "embed target '${TARGET}' has not been initialized with rtcx_add_embed()")
  endif()

  if(NOT DEFINED ARG_COMPRESSION)
    message(FATAL_ERROR "COMPRESSION argument is required")
  endif()

  if(NOT ARG_COMPRESSION STREQUAL "none" AND NOT ARG_COMPRESSION STREQUAL "zstd")
    message(FATAL_ERROR "COMPRESSION argument must be either none or zstd")
  endif()

  if(NOT DEFINED ARG_OUTPUT_DIRECTORY)
    message(FATAL_ERROR "OUTPUT_DIRECTORY argument is required")
  endif()

  get_property(EMBED_SOURCE_FILES TARGET ${TARGET}__embed_props PROPERTY EMBED_SOURCE_FILES)
  if(NOT EMBED_SOURCE_FILES)
    message(FATAL_ERROR "No source files registered for target '${TARGET}'")
  endif()

  get_property(EMBED_SOURCE_FILE_IDS TARGET ${TARGET}__embed_props PROPERTY EMBED_SOURCE_FILE_IDS)
  get_property(EMBED_SOURCE_FILE_DESTS TARGET ${TARGET}__embed_props
               PROPERTY EMBED_SOURCE_FILE_DESTS)
  get_property(EMBED_TARGET_DEPS TARGET ${TARGET}__embed_props PROPERTY EMBED_TARGET_DEPS)
  get_property(EMBED_TARGET_DEP_NAMES TARGET ${TARGET}__embed_props PROPERTY EMBED_TARGET_DEP_NAMES)
  get_property(EMBED_ARRAY_IDS TARGET ${TARGET}__embed_props PROPERTY EMBED_ARRAY_IDS)
  get_property(EMBED_ARRAY_VALUES TARGET ${TARGET}__embed_props PROPERTY EMBED_ARRAY_VALUES)
  get_property(EMBED_INCLUDE_DIRS TARGET ${TARGET}__embed_props PROPERTY EMBED_INCLUDE_DIRECTORIES)

  set(OUTPUT_DIR "${ARG_OUTPUT_DIRECTORY}")
  set(EMBED_SCRIPT_TEMPLATE "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/embed.in.cpp")
  set(CONFIGURED_EMBED_SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/${TARGET}__embed_cfg.cpp")
  set(EMBED_SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/${TARGET}__embed.cpp")

  set(EMBED_SCRIPT__ID "${TARGET}")
  set(EMBED_SCRIPT__ARRAY_IDS "${EMBED_ARRAY_IDS}")
  set(EMBED_SCRIPT__ARRAY_VALUES "${EMBED_ARRAY_VALUES}")
  set(EMBED_SCRIPT__FILE_IDS "${EMBED_SOURCE_FILE_IDS}")
  set(EMBED_SCRIPT__FILE_PATHS "${EMBED_SOURCE_FILES}")
  set(EMBED_SCRIPT__FILE_DESTS "${EMBED_SOURCE_FILE_DESTS}")
  set(EMBED_SCRIPT__INCLUDE_DIRS "${EMBED_INCLUDE_DIRS}")
  set(EMBED_SCRIPT__COMPRESSION "${ARG_COMPRESSION}")
  set(EMBED_SCRIPT__OUTPUT_DIR "${OUTPUT_DIR}")

  configure_file(${EMBED_SCRIPT_TEMPLATE} ${CONFIGURED_EMBED_SCRIPT} @ONLY)
  file(GENERATE OUTPUT "${EMBED_SCRIPT}" INPUT "${CONFIGURED_EMBED_SCRIPT}")

  set(RUNNER "${TARGET}__jit_embed_run")
  add_executable(${RUNNER} EXCLUDE_FROM_ALL
                 "${EMBED_SCRIPT}" ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../../../src/hash.cpp)
  target_link_libraries(${RUNNER} PRIVATE ${CMAKE_DL_LIBS} xxhash zstd)
  target_include_directories(${RUNNER} PRIVATE ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../../../include
                                               ${ZSTD_INCLUDE_DIR})
  set_target_properties(${RUNNER} PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES)

  add_custom_command(OUTPUT ${OUTPUT_DIR}/${TARGET}.hpp ${OUTPUT_DIR}/${TARGET}.s
                            ${OUTPUT_DIR}/${TARGET}.bin
                     COMMAND "${CMAKE_COMMAND}" -E env $<TARGET_FILE:${RUNNER}>
                     DEPENDS "${EMBED_SCRIPT}" ${EMBED_SOURCE_FILES} ${EMBED_TARGET_DEPS}
                             ${EMBED_TARGET_DEP_NAMES}
                     WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
                     COMMENT "Generating JIT embed for ${TARGET} into ${OUTPUT_DIR}"
                     VERBATIM)

  add_custom_target(${TARGET} ALL DEPENDS ${OUTPUT_DIR}/${TARGET}.hpp ${OUTPUT_DIR}/${TARGET}.s
                                          ${OUTPUT_DIR}/${TARGET}.bin
                    COMMENT "Custom target for JIT embed of ${TARGET}")

  message(STATUS "JIT embed for target ${TARGET} will be generated into: ${OUTPUT_DIR}/${TARGET}.hpp ${OUTPUT_DIR}/${TARGET}.s ${OUTPUT_DIR}/${TARGET}.bin"
  )

  set(${TARGET}_INCLUDE_DIRS "${OUTPUT_DIR}" PARENT_SCOPE)

  set(${TARGET}_SOURCE_DIR ${OUTPUT_DIR} PARENT_SCOPE)
endfunction()
