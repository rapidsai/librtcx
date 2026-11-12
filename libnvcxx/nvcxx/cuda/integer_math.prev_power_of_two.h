/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <nvcxx/cuda/std/bit.bit_floor.h>

#include <nvcxx/cuda/detail/math_integer>

namespace nvcxx {

template <detail::math_integer T>
constexpr T prev_power_of_two(T value)
{
  if (detail::negative(value)) { __builtin_unreachable(); }

  return static_cast<T>(std::bit_floor(static_cast<__make_unsigned(T)>(value)));
}

}  // namespace nvcxx
