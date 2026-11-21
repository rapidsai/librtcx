/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "test_support.cuh"

#include <cuda/std/chrono>
#include <cuda/std/type_traits>

template <class... T>
concept has_common_type = requires { typename traits::common_type_t<T...>; };

static_assert(traits::is_same_v<traits::common_type_t<char>, char>);
static_assert(traits::is_same_v<traits::common_type_t<int const&>, int>);
static_assert(traits::is_same_v<traits::common_type_t<int const&, double>, double>);
static_assert(traits::is_same_v<traits::common_type_t<char, unsigned char>, int>);
static_assert(traits::is_same_v<traits::common_type_t<int[2], int*>, int*>);
static_assert(traits::is_same_v<traits::common_type_t<int const[2], int*>, int const*>);
static_assert(traits::is_same_v<traits::common_type_t<short, int, long, double>, double>);
static_assert(traits::is_same_v<traits::common_type_t<void const, void volatile>, void>);
static_assert(traits::is_same_v<typename traits::common_type<int, long>::type, long>);
using function = int(double);
static_assert(traits::is_same_v<traits::common_type_t<function&, function*>, function*>);

struct unrelated {};
static_assert(!has_common_type<>);
static_assert(!has_common_type<void, int>);
static_assert(!has_common_type<int*, double*>);
static_assert(!has_common_type<unrelated, int>);
static_assert(!has_common_type<int, double, unrelated>);

struct lvalue_conversion {
  operator int() const&;
  operator int() && = delete;
};
static_assert(traits::is_same_v<traits::common_type_t<lvalue_conversion, int>, int>);

struct source {};
struct target {};
struct shared {};
template <>
struct traits::common_type<source, target> {
  using type = shared;
};
template <>
struct traits::common_type<shared, int> {
  using type = double;
};
static_assert(traits::is_same_v<traits::common_type_t<source, target>, shared>);
static_assert(traits::is_same_v<traits::common_type_t<source const&, target&&>, shared>);
static_assert(traits::is_same_v<traits::common_type_t<source, target, int>, double>);

namespace chrono      = traits::chrono;
using duration_result = traits::common_type_t<chrono::seconds, chrono::milliseconds>;
static_assert(traits::is_same_v<duration_result, chrono::milliseconds>);
struct test_clock {};
using seconds_point = chrono::time_point<test_clock, chrono::seconds>;
using millis_point  = chrono::time_point<test_clock, chrono::milliseconds>;
static_assert(traits::is_same_v<traits::common_type_t<seconds_point, millis_point>, millis_point>);
