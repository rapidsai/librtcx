/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <nvcxx/cuda/detail/bit_unsigned>

namespace nvcxx::std {

template <detail::bit_unsigned T>
constexpr int countr_zero(T x)
{
  if (x == 0) { return sizeof(T) * 8; }
  if (!__builtin_is_constant_evaluated()) {
    if constexpr (sizeof(T) <= 4) {
      return __ffs(static_cast<unsigned int>(x)) - 1;
    } else if constexpr (sizeof(T) <= 8) {
      return __ffsll(static_cast<unsigned long long>(x)) - 1;
    } else {
      auto low = static_cast<unsigned long long>(x);
      return low ? __ffsll(low) - 1 : 64 + __ffsll(static_cast<unsigned long long>(x >> 64)) - 1;
    }
  }
  int count = 0;
  while ((x & T{1}) == 0) {
    ++count;
    x >>= 1;
  }
  return count;
}

}  // namespace nvcxx::std
