/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <nvcxx/cuda/detail/bit_unsigned>

namespace nvcxx::std {

template <detail::bit_unsigned T>
constexpr int popcount(T x)
{
  if (!__builtin_is_constant_evaluated()) {
    if constexpr (sizeof(T) <= 4) {
      return __popc(static_cast<unsigned int>(x));
    } else if constexpr (sizeof(T) <= 8) {
      return __popcll(static_cast<unsigned long long>(x));
    } else {
      return __popcll(static_cast<unsigned long long>(x)) +
             __popcll(static_cast<unsigned long long>(x >> 64));
    }
  }
  int count = 0;
  while (x != 0) {
    ++count;
    x &= x - 1;
  }
  return count;
}

}  // namespace nvcxx::std
