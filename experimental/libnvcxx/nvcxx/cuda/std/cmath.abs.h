/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <nvcxx/cuda/detail/config>

namespace nvcxx::std {

template <class T>
constexpr T abs(T x)
{
  return x < T(0) ? -x : x;
}

}  // namespace nvcxx::std
