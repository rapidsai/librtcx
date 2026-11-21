/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <nvcxx/cuda/std/bit.bit_ceil.h>

#include <nvcxx/cuda/detail/integer_bounds>
#include <nvcxx/cuda/detail/math_integer>

namespace nvcxx {

template <detail::math_integer T>
constexpr T next_power_of_two(T value)
{
  if (detail::negative(value)) { __builtin_unreachable(); }

  auto result = std::bit_ceil(static_cast<__make_unsigned(T)>(value));
  if (result > static_cast<__make_unsigned(T)>(detail::integer_bounds<T>::max())) {
    __builtin_unreachable();
  }
  return static_cast<T>(result);
}

}  // namespace nvcxx
