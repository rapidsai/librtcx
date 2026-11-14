/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <nvcxx/cuda/std/type_traits>

namespace nvcxx::std {

using ::exp;

template <class T>
  requires is_integral_v<T>
__device__ inline double exp(T x)
{
  return ::exp(static_cast<double>(x));
}

}  // namespace nvcxx::std
