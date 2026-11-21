/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <nvcxx/cuda/detail/math_integer>

namespace nvcxx {

template <detail::math_integer T>
constexpr __make_unsigned(T) abs_diff(T a, T b)
{
  using U = __make_unsigned(T);
  return a < b ? static_cast<U>(static_cast<U>(b) - static_cast<U>(a))
               : static_cast<U>(static_cast<U>(a) - static_cast<U>(b));
}

}  // namespace nvcxx
