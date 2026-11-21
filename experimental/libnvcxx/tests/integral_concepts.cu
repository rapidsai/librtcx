/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "test_support.cuh"

#include <cuda/std/concepts>
#include <cuda/std/type_traits>

template <class T, bool Integral, bool Signed, bool Unsigned>
constexpr bool check_type()
{
  static_assert(traits::is_integral_v<T> == Integral);
  static_assert(traits::is_integral<T>::value == Integral);
  static_assert(traits::is_signed_v<T> == Signed);
  static_assert(traits::is_signed<T>::value == Signed);
  static_assert(traits::is_unsigned_v<T> == Unsigned);
  static_assert(traits::is_unsigned<T>::value == Unsigned);
  static_assert(traits::integral<T> == Integral);
  static_assert(traits::signed_integral<T> == (Integral && Signed));
  static_assert(traits::unsigned_integral<T> == (Integral && Unsigned));
  return true;
}

template <class T, bool Integral, bool Signed, bool Unsigned>
constexpr bool check_cv()
{
  return check_type<T, Integral, Signed, Unsigned>() &&
         check_type<T const, Integral, Signed, Unsigned>() &&
         check_type<T volatile, Integral, Signed, Unsigned>() &&
         check_type<T const volatile, Integral, Signed, Unsigned>();
}

static_assert(check_cv<bool, true, false, true>());
static_assert(check_cv<char, true, (char(-1) < char(0)), (char(-1) > char(0))>());
static_assert(check_cv<wchar_t, true, (wchar_t(-1) < wchar_t(0)), (wchar_t(-1) > wchar_t(0))>());
static_assert(check_cv<char8_t, true, false, true>());
static_assert(check_cv<char16_t, true, false, true>());
static_assert(check_cv<char32_t, true, false, true>());
static_assert(check_cv<signed char, true, true, false>());
static_assert(check_cv<short, true, true, false>());
static_assert(check_cv<int, true, true, false>());
static_assert(check_cv<long, true, true, false>());
static_assert(check_cv<long long, true, true, false>());
static_assert(check_cv<__int128, true, true, false>());
static_assert(check_cv<unsigned char, true, false, true>());
static_assert(check_cv<unsigned short, true, false, true>());
static_assert(check_cv<unsigned int, true, false, true>());
static_assert(check_cv<unsigned long, true, false, true>());
static_assert(check_cv<unsigned long long, true, false, true>());
static_assert(check_cv<unsigned __int128, true, false, true>());
static_assert(check_cv<float, false, true, false>());
static_assert(check_cv<double, false, true, false>());
static_assert(check_cv<long double, false, true, false>());

enum enumeration : unsigned { enumerator };
enum class scoped_enumeration : int { enumerator };
struct incomplete;
struct convertible {
  operator int() const;
};

static_assert(check_cv<void, false, false, false>());
static_assert(check_cv<int*, false, false, false>());
static_assert(check_cv<int[3], false, false, false>());
static_assert(check_cv<enumeration, false, false, false>());
static_assert(check_cv<scoped_enumeration, false, false, false>());
static_assert(check_cv<incomplete, false, false, false>());
static_assert(check_cv<convertible, false, false, false>());
static_assert(check_type<int&, false, false, false>());
static_assert(check_type<unsigned&&, false, false, false>());
static_assert(check_type<void(), false, false, false>());

template <traits::integral T>
constexpr int select_integral(T)
{
  return 0;
}

template <traits::signed_integral T>
constexpr int select_integral(T)
{
  return 1;
}

template <traits::unsigned_integral T>
constexpr int select_integral(T)
{
  return 2;
}

static_assert(select_integral(1) == 1);
static_assert(select_integral(1u) == 2);
static_assert(select_integral(true) == 2);

#ifdef NVCXX_AS_STD
static_assert(cuda::std::signed_integral<int>);
static_assert(cuda::std::unsigned_integral<unsigned>);
static_assert(std::integral<bool>);
#endif
