/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <nvcxx/cuda/std/bit.countl_zero.h>

#include <nvcxx/cuda/detail/bit_unsigned>

namespace nvcxx::std {

template <detail::bit_unsigned T>
constexpr int bit_width(T x)
{
  return int(sizeof(T) * 8) - countl_zero(x);
}

}  // namespace nvcxx::std
