/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <nvcxx/cuda/detail/math_integer>

namespace nvcxx {

template <detail::math_integer T, detail::math_integer U>
constexpr std::common_type_t<T, U> round_down(T x, U y)
{
  if (detail::negative(x) || y <= 0) { __builtin_unreachable(); }

  using R = std::common_type_t<T, U>;
  using W = __make_unsigned(decltype(x / y));
  W a     = static_cast<W>(x);
  W b     = static_cast<W>(y);
  return static_cast<R>(a - a % b);
}

}  // namespace nvcxx
