/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <nvcxx/cuda/detail/bit_unsigned>

namespace nvcxx {

template <std::detail::bit_unsigned T>
constexpr T bit_reverse(T value)
{
  if (!__builtin_is_constant_evaluated()) {
    if constexpr (sizeof(T) <= 4) {
      return static_cast<T>(__brev(static_cast<unsigned int>(value)) >> (32 - sizeof(T) * 8));
    } else if constexpr (sizeof(T) == 8) {
      return static_cast<T>(__brevll(static_cast<unsigned long long>(value)));
    } else {
      return (T(__brevll(static_cast<unsigned long long>(value))) << 64) |
             T(__brevll(static_cast<unsigned long long>(value >> 64)));
    }
  }
  T result = 0;
  for (int i = 0; i < int(sizeof(T) * 8); ++i) {
    result = static_cast<T>((result << 1) | (value & T(1)));
    value >>= 1;
  }
  return result;
}

}  // namespace nvcxx
