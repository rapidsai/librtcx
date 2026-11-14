/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <nvcxx/cuda/std/type_traits>

namespace nvcxx::std {

using ::sinh;

template <class T>
  requires is_integral_v<T>
__device__ inline double sinh(T x)
{
  return ::sinh(static_cast<double>(x));
}

}  // namespace nvcxx::std
