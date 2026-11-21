/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <nvcxx/cuda/integer_math.ilog2.h>

#include <nvcxx/cuda/detail/math_integer>

namespace nvcxx {

template <detail::math_integer T>
constexpr int ceil_ilog2(T value)
{
  if (value <= 0) { __builtin_unreachable(); }

  return value == 1 ? 0 : ilog2(static_cast<T>(value - 1)) + 1;
}

}  // namespace nvcxx
