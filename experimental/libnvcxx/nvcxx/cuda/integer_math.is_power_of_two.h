/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <nvcxx/cuda/detail/math_integer>

namespace nvcxx {

template <detail::math_integer T>
constexpr bool is_power_of_two(T value)
{
  if (detail::negative(value)) { __builtin_unreachable(); }

  using U = __make_unsigned(T);
  U x     = static_cast<U>(value);
  return x != 0 && (x & static_cast<U>(x - 1)) == 0;
}

}  // namespace nvcxx
