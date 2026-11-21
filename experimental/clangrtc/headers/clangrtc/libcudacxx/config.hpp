/* SPDX-License-Identifier: Apache-2.0 */

/**
 * @file config.hpp
 * @brief Require libcudacxx's explicit freestanding configuration for std forwarding.
 */
#pragma once

#include <cuda/std/detail/__config>

#ifndef _CCCL_FREESTANDING
#error "--stdlib=libcudacxx requires CCCL with _CCCL_ENABLE_FREESTANDING support; supply its include root with -I"
#elif !_CCCL_FREESTANDING()
#error "--stdlib=libcudacxx requires _CCCL_ENABLE_FREESTANDING"
#endif
