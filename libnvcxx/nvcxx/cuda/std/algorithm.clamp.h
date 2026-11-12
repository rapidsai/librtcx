/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <nvcxx/cuda/std/algorithm.max.h>
#include <nvcxx/cuda/std/algorithm.min.h>

namespace nvcxx::std {

template <class T>
constexpr T const& clamp(T const& x, T const& lo, T const& hi)
{
  return min(max(x, lo), hi);
}

}  // namespace nvcxx::std
