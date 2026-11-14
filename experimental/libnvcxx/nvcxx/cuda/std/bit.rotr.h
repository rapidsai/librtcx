/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <nvcxx/cuda/detail/bit_unsigned>

namespace nvcxx::std {

template <detail::bit_unsigned T>
constexpr T rotr(T x, int shift)
{
  constexpr int width = sizeof(T) * 8;
  int rotation        = shift % width;
  if (rotation < 0) { rotation += width; }
  if (rotation == 0) { return x; }
  return static_cast<T>((x >> rotation) | (x << (width - rotation)));
}

}  // namespace nvcxx::std
