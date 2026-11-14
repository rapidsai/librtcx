/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <nvcxx/cuda/std/type_traits>

namespace nvcxx::std {

using ::floor;

template <class T>
  requires is_integral_v<T>
__device__ inline double floor(T x)
{
  return ::floor(static_cast<double>(x));
}

}  // namespace nvcxx::std
