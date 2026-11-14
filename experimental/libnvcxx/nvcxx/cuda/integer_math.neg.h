/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <nvcxx/cuda/detail/math_integer>

namespace nvcxx {

template <detail::math_integer T>
constexpr T neg(T value)
{
  using U = __make_unsigned(T);
  return static_cast<T>(U(0) - static_cast<U>(value));
}

}  // namespace nvcxx
