/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <nvcxx/cuda/std/type_traits>

namespace nvcxx::std {

using ::trunc;

template <class T>
  requires is_integral_v<T>
__device__ inline double trunc(T x)
{
  return ::trunc(static_cast<double>(x));
}

}  // namespace nvcxx::std
