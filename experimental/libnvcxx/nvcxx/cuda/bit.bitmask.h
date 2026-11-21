/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <nvcxx/cuda/detail/config>

namespace nvcxx {

template <class T>
constexpr T bitmask(int start, int width)
{
  return width == int(sizeof(T) * 8) ? T(-1) : ((T(1) << width) - 1) << start;
}

}  // namespace nvcxx
