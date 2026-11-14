/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <nvcxx/cuda/detail/math_integer>

namespace nvcxx {

template <detail::math_integer T, detail::math_integer E>
constexpr T ipow(T base, E exponent)
{
  if (base == 0 && detail::negative(exponent)) { __builtin_unreachable(); }

  if (exponent == 0 || base == 1) { return T(1); }

  if (detail::negative(exponent) || base == 0) { return T(0); }

  using U = __make_unsigned(T);

  U power  = static_cast<U>(base);
  U result = 1;

  while (exponent != 0) {
    if (exponent & 1) result = static_cast<U>(static_cast<unsigned __int128>(result) * power);
    exponent = static_cast<E>(exponent >> 1);
    if (exponent != 0) power = static_cast<U>(static_cast<unsigned __int128>(power) * power);
  }

  return static_cast<T>(result);
}

}  // namespace nvcxx
