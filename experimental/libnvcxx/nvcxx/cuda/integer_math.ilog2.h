/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <nvcxx/cuda/std/bit.bit_width.h>

#include <nvcxx/cuda/detail/math_integer>

namespace nvcxx {

template <detail::math_integer T>
constexpr int ilog2(T value)
{
  if (value <= 0) { __builtin_unreachable(); }

  return std::bit_width(static_cast<__make_unsigned(T)>(value)) - 1;
}

}  // namespace nvcxx
