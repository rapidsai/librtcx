/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <nvcxx/cuda/detail/math_integer>

namespace nvcxx {

template <detail::math_integer T>
constexpr int ilog10(T value)
{
  if (value <= 0) { __builtin_unreachable(); }

  int result = 0;
  while (value >= 10) {
    value = static_cast<T>(value / 10);
    ++result;
  }
  return result;
}

}  // namespace nvcxx
