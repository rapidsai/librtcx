/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <nvcxx/cuda/std/bit.bit_width.h>

#include <nvcxx/cuda/detail/bit_unsigned>

namespace nvcxx::std {

template <detail::bit_unsigned T>
constexpr T bit_ceil(T x)
{
  if (x <= 1) { return T(1); }
  int shift = bit_width(static_cast<T>(x - 1));
  if (shift == int(sizeof(T) * 8)) { __builtin_unreachable(); }
  return static_cast<T>(T(1) << shift);
}

}  // namespace nvcxx::std
