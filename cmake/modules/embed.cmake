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

CMake module providing functions to embed source files, headers, and binary
blobs into executables for JIT compilation at runtime.

The embed system is built around four functions that must be called in order:

1. :ref:`rtcx_add_embed` - initialize an embed target.
2. :ref:`rtcx_embed_includes` - register header/source directories (call 0 or more times).
3. :ref:`rtcx_embed_blob` - register binary blobs or compiled objects (call 0 or more times).
4. :ref:`rtcx_embed` - finalize and generate the embedded artifacts.

Each function is defined in a separate file under ``embed/`` and auto-included
when this file is included.

.. note::

  The ``zstd`` and ``xxhash`` CMake targets must be available before calling
  ``rtcx_embed``.

#]=======================================================================]

include(${CMAKE_CURRENT_LIST_DIR}/embed/add_embed.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/embed/embed_includes.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/embed/embed_blob.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/embed/embed.cmake)
