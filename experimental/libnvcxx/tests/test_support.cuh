/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuda/std/type_traits>

#ifdef TEST_LIBCUDACXX
namespace traits = cuda::std;
#else
namespace traits = nvcxx::std;
#endif

template <class... T>
struct test_types {};

template <class... T, class F>
constexpr bool for_types(test_types<T...>, F check)
{
  return (check.template operator()<T>() && ...);
}

#define CHECK_TRAIT(name, expected, ...)                      \
  static_assert(traits::name##_v<__VA_ARGS__> == (expected)); \
  static_assert(traits::name<__VA_ARGS__>::value == (expected))

#define CHECK_CONCEPT(name, expected, ...) static_assert(traits::name<__VA_ARGS__> == (expected))
