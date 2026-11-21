/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "test_support.cuh"

#include <cuda/std/chrono>
#include <cuda/std/cmath>

#ifndef TEST_LIBCUDACXX
#include <cuda/integer_math>
#endif

namespace chrono = traits::chrono;
#ifdef TEST_LIBCUDACXX
template <long long Num, long long Den = 1>
using period = traits::ratio<Num, Den>;
#else
template <long long Num, long long Den = 1>
using period = traits::rational<Num, Den>;
#endif

// Mixed representations and periods still select the correct result without public traits.
constexpr auto sum = chrono::seconds{2} + chrono::milliseconds{500};
static_assert(sum.count() == 2500);
static_assert(traits::is_same_v<traits::remove_cv_t<decltype(sum)>, chrono::milliseconds>);
static_assert((chrono::seconds{2} - chrono::milliseconds{500}).count() == 1500);
static_assert(chrono::seconds{2} == chrono::milliseconds{2000});
static_assert(chrono::milliseconds{999} < chrono::seconds{1});
static_assert(chrono::seconds{5} / chrono::milliseconds{500} == 10);
static_assert((chrono::milliseconds{2500} % chrono::seconds{1}).count() == 500);
static_assert((chrono::seconds{7} % 3).count() == 1);
static_assert((chrono::milliseconds{3} * 2.5).count() == 7.5);
static_assert((2 * chrono::seconds{3}).count() == 6);
static_assert((chrono::seconds{3} / 2.0).count() == 1.5);
using thirds            = chrono::duration<int, period<1, 3>>;
using fifths            = chrono::duration<long long, period<1, 5>>;
constexpr auto fraction = thirds{1} + fifths{1};
static_assert(fraction.count() == 8);
static_assert(decltype(fraction)::period::num == 1 && decltype(fraction)::period::den == 15);
static_assert(traits::is_same_v<typename decltype(fraction)::rep, long long>);
static_assert(chrono::duration_cast<chrono::seconds>(chrono::milliseconds{-1500}).count() == -1);
static_assert(chrono::duration_cast<chrono::duration<double>>(chrono::milliseconds{1500}).count() ==
              1.5);

struct test_clock {};
using seconds_point  = chrono::time_point<test_clock, chrono::seconds>;
constexpr auto point = seconds_point{chrono::seconds{2}} + chrono::milliseconds{500};
static_assert(point.time_since_epoch().count() == 2500);
static_assert((point - seconds_point{chrono::seconds{1}}).count() == 1500);
static_assert((point - chrono::milliseconds{500}) == seconds_point{chrono::seconds{2}});
static_assert(chrono::time_point_cast<chrono::seconds>(point).time_since_epoch().count() == 2);

#ifdef LIBNVCXX
// Extended integer representations retain the wide conversion intermediate.
using wide_seconds = chrono::duration<__int128>;
using wide_nanos   = chrono::duration<__int128, period<1, 1000000000>>;
static_assert(chrono::duration_cast<wide_nanos>(wide_seconds{9000000000000000LL}).count() ==
              __int128(9000000000000000LL) * 1000000000);
using unsigned_millis = chrono::duration<unsigned __int128, period<1, 1000>>;
static_assert(chrono::duration_cast<unsigned_millis>(chrono::seconds{3}).count() == 3000);

static_assert(nvcxx::ceil_div(7, 3L) == 3);
static_assert(nvcxx::round_down(7, 3L) == 6);
static_assert(nvcxx::round_up(7, 3L) == 9);
static_assert(traits::is_same_v<decltype(nvcxx::ceil_div(short(7), short(3))), short>);
static_assert(traits::is_same_v<decltype(nvcxx::round_down(7, 3L)), long>);
static_assert(traits::is_same_v<decltype(nvcxx::round_up(7u, 3)), unsigned>);
static_assert(traits::is_same_v<decltype(nvcxx::ceil_div(__int128(7), 3)), __int128>);
#endif

// Integer arguments promote to double; all-float overloads preserve float.
static_assert(traits::is_same_v<decltype(traits::atan2(1.0f, 2.0f)), float>);
static_assert(traits::is_same_v<decltype(traits::atan2(1, 2.0f)), double>);
static_assert(traits::is_same_v<decltype(traits::hypot(1.0f, 2.0)), double>);
static_assert(traits::is_same_v<decltype(traits::fma(1.0f, 2.0f, 3.0f)), float>);
static_assert(traits::is_same_v<decltype(traits::fma(1.0f, 2, 3.0f)), double>);
static_assert(traits::is_same_v<decltype(traits::hypot(1.0f, 2.0f, 3.0f)), float>);
static_assert(traits::is_same_v<decltype(traits::hypot(1.0f, 2.0, 3.0f)), double>);

extern "C" __global__ void mixed_math_overloads(double* result)
{
  *result = traits::atan2(1, 2.0f) + traits::hypot(3.0f, 4) + traits::fma(2, 3.0f, 4.0f);
}
