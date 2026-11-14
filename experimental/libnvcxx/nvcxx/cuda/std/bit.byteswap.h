/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <nvcxx/cuda/std/type_traits>

namespace nvcxx::std {

template <class T>
  requires is_integral_v<T>
constexpr T byteswap(T x)
{
  if constexpr (sizeof(T) == 1) {
    return x;
  } else {
    using U = __make_unsigned(T);
    U value = __builtin_bit_cast(U, x);
    if (!__builtin_is_constant_evaluated()) {
      if constexpr (sizeof(T) == 2) {
        return __builtin_bit_cast(
          T, static_cast<U>(__byte_perm(static_cast<unsigned int>(value), 0, 0x4401)));
      } else if constexpr (sizeof(T) == 4) {
        return __builtin_bit_cast(
          T, static_cast<U>(__byte_perm(static_cast<unsigned int>(value), 0, 0x0123)));
      } else if constexpr (sizeof(T) == 8) {
        U low  = __byte_perm(static_cast<unsigned int>(value), 0, 0x0123);
        U high = __byte_perm(static_cast<unsigned int>(value >> 32), 0, 0x0123);
        return __builtin_bit_cast(T, static_cast<U>((low << 32) | high));
      } else {
        U low  = byteswap(static_cast<unsigned long long>(value));
        U high = byteswap(static_cast<unsigned long long>(value >> 64));
        return __builtin_bit_cast(T, static_cast<U>((low << 64) | high));
      }
    }
    U result = 0;
    for (int i = 0; i < int(sizeof(T)); ++i) {
      result = static_cast<U>((result << 8) | (value & U(255)));
      value >>= 8;
    }
    return __builtin_bit_cast(T, result);
  }
}

}  // namespace nvcxx::std
