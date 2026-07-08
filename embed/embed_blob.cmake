# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include_guard(GLOBAL)

function(rtcx_embed_blob TARGET)
  set(OPTIONS)
  set(ONE_VALUE_ARGS ID FILE DEST)
  set(MULTI_VALUE_ARGS ARRAY_IDS ARRAY_VALUES)
  cmake_parse_arguments(ARG "${OPTIONS}" "${ONE_VALUE_ARGS}" "${MULTI_VALUE_ARGS}" ${ARGN})

  if(NOT TARGET ${TARGET}__embed_props)
    message(FATAL_ERROR "embed target '${TARGET}' has not been initialized with rtcx_add_embed()")
  endif()

  if(NOT ARG_ID
     OR NOT ARG_FILE
     OR NOT ARG_DEST
  )
    message(FATAL_ERROR "ID, FILE, and DEST arguments are required")
  endif()

  if(ARG_ARRAY_IDS)
    if(NOT ARG_ARRAY_VALUES)
      message(FATAL_ERROR "ARRAY_VALUES argument is required when ARRAY_IDS is provided")
    endif()

    list(LENGTH ARG_ARRAY_IDS ARG_ARRAY_IDS_LENGTH)
    list(LENGTH ARG_ARRAY_VALUES ARG_ARRAY_VALUES_LENGTH)

    if(NOT ARG_ARRAY_IDS_LENGTH EQUAL ARG_ARRAY_VALUES_LENGTH)
      message(FATAL_ERROR "ARRAY_IDS and ARRAY_VALUES must have the same length")
    endif()

    set_property(
      TARGET ${TARGET}__embed_props
      APPEND
      PROPERTY EMBED_ARRAY_IDS ${ARG_ARRAY_IDS}
    )
    set_property(
      TARGET ${TARGET}__embed_props
      APPEND
      PROPERTY EMBED_ARRAY_VALUES ${ARG_ARRAY_VALUES}
    )
  endif()

  if(ARG_FILE MATCHES "\\$<TARGET_OBJECTS:([^>]+)>")
    # If the file is a generator expression for target objects add as dependency
    set_property(
      TARGET ${TARGET}__embed_props
      APPEND
      PROPERTY EMBED_TARGET_DEPS $<TARGET_OBJECTS:${CMAKE_MATCH_1}>
    )
    # Also record the target name itself. Depending only on $<TARGET_OBJECTS:...> creates file-level
    # dependencies without a target-level ordering, which breaks the Makefiles generator (Ninja
    # resolves it via its global build graph).
    set_property(
      TARGET ${TARGET}__embed_props
      APPEND
      PROPERTY EMBED_TARGET_DEP_NAMES ${CMAKE_MATCH_1}
    )
  else()
    if(NOT EXISTS "${ARG_FILE}")
      message(FATAL_ERROR "Source file '${ARG_FILE}' does not exist")
    endif()
  endif()

  set_property(
    TARGET ${TARGET}__embed_props
    APPEND
    PROPERTY EMBED_SOURCE_FILE_IDS ${ARG_ID}
  )
  set_property(
    TARGET ${TARGET}__embed_props
    APPEND
    PROPERTY EMBED_SOURCE_FILES ${ARG_FILE}
  )
  set_property(
    TARGET ${TARGET}__embed_props
    APPEND
    PROPERTY EMBED_SOURCE_FILE_DESTS ${ARG_DEST}
  )

  get_property(
    SOURCE_FILE_IDS
    TARGET ${TARGET}__embed_props
    PROPERTY EMBED_SOURCE_FILE_IDS
  )
  list(LENGTH SOURCE_FILE_IDS IDX)

  set_property(TARGET ${TARGET}__embed_props PROPERTY EMBED_FILE_INDEX ${IDX})
endfunction()
