/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <nvcxx/cuda/detail/config>

namespace nvcxx::std {

template <class T>
constexpr T const& max(T const& a, T const& b)
{
  return a < b ? b : a;
}

}  // namespace nvcxx::std
