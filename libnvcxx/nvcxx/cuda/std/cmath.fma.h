/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <nvcxx/cuda/detail/math_arithmetic>

namespace nvcxx::std {

using ::fma;

template <detail::math_arithmetic T, detail::math_arithmetic U, detail::math_arithmetic V>
__device__ inline detail::math_common_t<T, U, V> fma(T x, U y, V z)
{
  using R = detail::math_common_t<T, U, V>;
  return nvcxx::std::fma(static_cast<R>(x), static_cast<R>(y), static_cast<R>(z));
}

}  // namespace nvcxx::std
