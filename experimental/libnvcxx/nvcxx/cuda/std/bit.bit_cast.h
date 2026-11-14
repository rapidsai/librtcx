/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <nvcxx/cuda/detail/config>

namespace nvcxx::std {

template <class To, class From>
__device__ constexpr To bit_cast(From const& x)
{
  return __builtin_bit_cast(To, x);
}

}  // namespace nvcxx::std
