/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <nvcxx/cuda/detail/bit_unsigned>

namespace nvcxx::std {

template <detail::bit_unsigned T>
constexpr int countl_zero(T x)
{
  constexpr int width = sizeof(T) * 8;
  if (x == 0) { return width; }
  if (!__builtin_is_constant_evaluated()) {
    if constexpr (sizeof(T) <= 4) {
      return __clz(static_cast<unsigned int>(x)) - (32 - width);
    } else if constexpr (sizeof(T) <= 8) {
      return __clzll(static_cast<unsigned long long>(x));
    } else {
      auto high = static_cast<unsigned long long>(x >> 64);
      return high ? __clzll(high) : 64 + __clzll(static_cast<unsigned long long>(x));
    }
  }
  int count = 0;
  T mask    = T(1) << (width - 1);
  while ((x & mask) == 0) {
    ++count;
    mask >>= 1;
  }
  return count;
}

}  // namespace nvcxx::std
