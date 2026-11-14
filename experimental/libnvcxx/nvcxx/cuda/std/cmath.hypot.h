/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <nvcxx/cuda/detail/math_arithmetic>

namespace nvcxx::std {

using ::hypot;

template <detail::math_arithmetic T, detail::math_arithmetic U>
__device__ inline detail::math_common_t<T, U> hypot(T x, U y)
{
  using R = detail::math_common_t<T, U>;
  return nvcxx::std::hypot(static_cast<R>(x), static_cast<R>(y));
}

template <detail::math_arithmetic T, detail::math_arithmetic U, detail::math_arithmetic V>
__device__ inline detail::math_common_t<T, U, V> hypot(T x, U y, V z)
{
  using R = detail::math_common_t<T, U, V>;
  if constexpr (is_same_v<R, float>) {
    return ::norm3df(static_cast<R>(x), static_cast<R>(y), static_cast<R>(z));
  } else {
    return ::norm3d(static_cast<R>(x), static_cast<R>(y), static_cast<R>(z));
  }
}

}  // namespace nvcxx::std
