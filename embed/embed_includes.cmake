# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include_guard(GLOBAL)

#[=======================================================================[.rst:
rtcx_embed_includes
--------------------

Register a directory of source and header files into an embed target for JIT embedding.

.. code-block:: cmake

  rtcx_embed_includes(<target>
      SOURCE_DIRECTORY <dir>
      DEST_DIRECTORY <dest>
      INCLUDE_DIRECTORIES <dir>...
      [FILES <file>...])

Registers files from ``SOURCE_DIRECTORY`` into embed target ``<target>``. May be
called multiple times on the same target. Must be called after ``rtcx_add_embed``
and before ``rtcx_embed``.

``<target>``
  Required. Name of the embed target, previously initialized with ``rtcx_add_embed``.

``SOURCE_DIRECTORY``
  Required. Directory containing the source files to embed.

``DEST_DIRECTORY``
  Required. Virtual destination path within the embedded filesystem.

``INCLUDE_DIRECTORIES``
  Required. Include paths that will be available when compiling with these
  embedded files.

``FILES``
  Optional. List of files relative to ``SOURCE_DIRECTORY`` to register. If
  omitted, all files under ``SOURCE_DIRECTORY`` are globbed recursively.
#]=======================================================================]

function(rtcx_embed_includes TARGET)
  set(OPTIONS "")
  set(ONE_VALUE_ARGS SOURCE_DIRECTORY # Source directory where files will be copied from
                     DEST_DIRECTORY # Destination directory where files will be copied to
  )
  set(MULTI_VALUE_ARGS
      FILES # Source files relative to SOURCE_DIRECTORY (optional, if not provided, all files under
            # SOURCE_DIRECTORY will be used)
      INCLUDE_DIRECTORIES # Include directories to be used when compiling with these files
  )
  cmake_parse_arguments(ARG "${OPTIONS}" "${ONE_VALUE_ARGS}" "${MULTI_VALUE_ARGS}" ${ARGN})

  if(NOT TARGET ${TARGET}__embed_props)
    message(FATAL_ERROR "embed target '${TARGET}' has not been initialized with add_embed()")
  endif()

  if(NOT ARG_SOURCE_DIRECTORY
     OR NOT ARG_DEST_DIRECTORY
     OR NOT ARG_INCLUDE_DIRECTORIES
  )
    message(
      FATAL_ERROR "SOURCE_DIRECTORY, DEST_DIRECTORY, and INCLUDE_DIRECTORIES arguments are required"
    )
  endif()

  if(NOT ARG_FILES)
    # gather all include files under the specified directory
    file(GLOB_RECURSE INCLUDE_FILES "${ARG_SOURCE_DIRECTORY}/*")

    # get their paths relative to the base include directory
    set(INCLUDE_FILES_RELATIVE_PATHS "")
    foreach(INCLUDE_FILE IN LISTS INCLUDE_FILES)
      file(RELATIVE_PATH INCLUDE_FILE_REL_PATH "${ARG_SOURCE_DIRECTORY}" "${INCLUDE_FILE}")
      list(APPEND INCLUDE_FILES_RELATIVE_PATHS "${INCLUDE_FILE_REL_PATH}")
    endforeach()

    set(ARG_FILES ${INCLUDE_FILES_RELATIVE_PATHS})
  endif()

  # check that each source file exists
  foreach(SOURCE_FILE IN LISTS ARG_FILES)
    if(NOT EXISTS "${ARG_SOURCE_DIRECTORY}/${SOURCE_FILE}")
      message(FATAL_ERROR "Source file '${ARG_SOURCE_DIRECTORY}/${SOURCE_FILE}' does not exist")
    endif()
  endforeach(SOURCE_FILE)

  # Determine the starting index for new IDs from the current list length
  get_property(
    SOURCE_FILE_IDS
    TARGET ${TARGET}__embed_props
    PROPERTY EMBED_SOURCE_FILE_IDS
  )
  list(LENGTH SOURCE_FILE_IDS IDX)

  foreach(SOURCE_FILE IN LISTS ARG_FILES)
    set_property(
      TARGET ${TARGET}__embed_props
      APPEND
      PROPERTY EMBED_SOURCE_FILE_IDS "include_${IDX}"
    )
    set_property(
      TARGET ${TARGET}__embed_props
      APPEND
      PROPERTY EMBED_SOURCE_FILES "${ARG_SOURCE_DIRECTORY}/${SOURCE_FILE}"
    )
    set_property(
      TARGET ${TARGET}__embed_props
      APPEND
      PROPERTY EMBED_SOURCE_FILE_DESTS "${ARG_DEST_DIRECTORY}/${SOURCE_FILE}"
    )
    math(EXPR IDX "${IDX} + 1")
  endforeach()

  set_property(
    TARGET ${TARGET}__embed_props
    APPEND
    PROPERTY EMBED_INCLUDE_DIRECTORIES ${ARG_INCLUDE_DIRECTORIES}
  )

  get_property(
    SOURCE_FILE_IDS
    TARGET ${TARGET}__embed_props
    PROPERTY EMBED_SOURCE_FILE_IDS
  )
  list(LENGTH SOURCE_FILE_IDS IDX)

  set_property(TARGET ${TARGET}__embed_props PROPERTY EMBED_FILE_INDEX ${IDX})
endfunction()
