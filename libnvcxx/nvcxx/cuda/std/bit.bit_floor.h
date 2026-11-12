/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <nvcxx/cuda/std/bit.bit_width.h>

#include <nvcxx/cuda/detail/bit_unsigned>

namespace nvcxx::std {

template <detail::bit_unsigned T>
constexpr T bit_floor(T x)
{
  return x == 0 ? T(0) : static_cast<T>(T(1) << (bit_width(x) - 1));
}

}  // namespace nvcxx::std
