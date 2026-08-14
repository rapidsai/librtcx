# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include_guard(GLOBAL)

#[=======================================================================[.rst:
rtcx_add_embed
---------------

Initialize an embed target for JIT embedding.

.. code-block:: cmake

  rtcx_add_embed(<target>)

Initializes the embed target ``<target>``. Must be called before
``rtcx_embed_includes``, ``rtcx_embed_blob``, or ``rtcx_embed``.

``<target>``
  Required. Name of the logical embed group to initialize.
#]=======================================================================]

function(rtcx_add_embed TARGET)
  set(OPTIONS "")
  set(ONE_VALUE_ARGS)
  set(MULTI_VALUE_ARGS)
  cmake_parse_arguments(ARG "${OPTIONS}" "${ONE_VALUE_ARGS}" "${MULTI_VALUE_ARGS}" ${ARGN})

  if(NOT DEFINED TARGET)
    message(FATAL_ERROR "TARGET argument is required")
  endif()

  add_library(${TARGET}__embed_props INTERFACE)
  set_property(TARGET ${TARGET}__embed_props PROPERTY EMBED_FILE_INDEX 0)
endfunction()
