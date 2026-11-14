/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <nvcxx/cuda/std/bit.countr_zero.h>

#include <nvcxx/cuda/detail/bit_unsigned>

namespace nvcxx::std {

template <detail::bit_unsigned T>
constexpr int countr_one(T x)
{
  return countr_zero(static_cast<T>(~x));
}

}  // namespace nvcxx::std
