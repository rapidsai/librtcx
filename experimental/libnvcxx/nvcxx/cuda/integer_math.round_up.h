/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <nvcxx/cuda/integer_math.ceil_div.h>

#include <nvcxx/cuda/detail/integer_bounds>
#include <nvcxx/cuda/detail/math_integer>

namespace nvcxx {

template <detail::math_integer T, detail::math_integer U>
constexpr std::common_type_t<T, U> round_up(T x, U y)
{
  using R       = std::common_type_t<T, U>;
  auto quotient = ceil_div(x, y);

  if (static_cast<unsigned __int128>(quotient) >
      static_cast<unsigned __int128>(detail::integer_bounds<R>::max()) /
        static_cast<unsigned __int128>(y)) {
    __builtin_unreachable();
  }

  return static_cast<R>(static_cast<unsigned __int128>(quotient) *
                        static_cast<unsigned __int128>(y));
}

}  // namespace nvcxx
