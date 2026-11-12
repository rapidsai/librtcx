/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <nvcxx/cuda/detail/config>

namespace nvcxx::std {

template <class T>
constexpr T const& min(T const& a, T const& b)
{
  return b < a ? b : a;
}

}  // namespace nvcxx::std
