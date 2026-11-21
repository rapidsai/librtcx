/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <nvcxx/cuda/detail/math_arithmetic>

namespace nvcxx::std {

using ::atan2;

template <detail::math_arithmetic T, detail::math_arithmetic U>
__device__ inline detail::math_common_t<T, U> atan2(T x, U y)
{
  using R = detail::math_common_t<T, U>;
  return nvcxx::std::atan2(static_cast<R>(x), static_cast<R>(y));
}

}  // namespace nvcxx::std
