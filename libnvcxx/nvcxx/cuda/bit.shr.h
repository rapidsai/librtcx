/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <nvcxx/cuda/detail/bit_unsigned>

namespace nvcxx {

template <std::detail::bit_unsigned T>
constexpr T shr(T value, int shift)
{
  return static_cast<unsigned int>(shift) >= sizeof(T) * 8 ? T(0) : static_cast<T>(value >> shift);
}

}  // namespace nvcxx
