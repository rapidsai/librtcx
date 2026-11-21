/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "test_support.cuh"

#include <cuda/std/type_traits>

template <class T>
constexpr bool check_transformations()
{
  static_assert(traits::is_same_v<traits::remove_cv_t<T const volatile>, T>);
  static_assert(traits::is_same_v<typename traits::remove_cv<T const volatile>::type, T>);
  static_assert(traits::is_same_v<traits::remove_cv_t<T const&>, T const&>);
  static_assert(traits::is_same_v<traits::remove_reference_t<T const&>, T const>);
  static_assert(traits::is_same_v<typename traits::remove_reference<T&&>::type, T>);
  static_assert(traits::is_same_v<traits::remove_cvref_t<T const volatile&&>, T>);
  static_assert(traits::is_same_v<traits::decay_t<T const volatile&>, T>);
  static_assert(traits::is_same_v<decltype(traits::declval<T>()), T&&>);
  static_assert(traits::is_same_v<decltype(traits::declval<T&>()), T&>);
  static_assert(traits::is_same_v<decltype(traits::declval<T const&&>()), T const&&>);
  return true;
}

struct incomplete;
struct record {};
static_assert(for_types(test_types<int, unsigned, __int128, double, int*, record, incomplete>{},
                        []<class T> { return check_transformations<T>(); }));
static_assert(traits::is_same_v<traits::remove_cv_t<void const volatile>, void>);
static_assert(traits::is_same_v<traits::remove_reference_t<void>, void>);
static_assert(traits::is_same_v<decltype(traits::declval<void>()), void>);
static_assert(traits::is_same_v<traits::remove_cvref_t<int const (&)[2]>, int[2]>);
static_assert(traits::is_same_v<traits::decay_t<int const (&)[2]>, int const*>);
static_assert(traits::is_same_v<traits::decay_t<int[2][3]>, int (*)[3]>);
using function = int(double);
static_assert(traits::is_same_v<traits::decay_t<function&>, function*>);
static_assert(traits::is_same_v<decltype(traits::declval<function>()), function&>);

template <class Signed, class Unsigned>
constexpr bool check_signed_transformations()
{
  static_assert(traits::is_same_v<traits::make_signed_t<Unsigned>, Signed>);
  static_assert(traits::is_same_v<typename traits::make_signed<Unsigned>::type, Signed>);
  static_assert(
    traits::is_same_v<traits::make_signed_t<Unsigned const volatile>, Signed const volatile>);
  static_assert(traits::is_same_v<traits::make_unsigned_t<Signed>, Unsigned>);
  static_assert(traits::is_same_v<typename traits::make_unsigned<Signed>::type, Unsigned>);
  static_assert(
    traits::is_same_v<traits::make_unsigned_t<Signed const volatile>, Unsigned const volatile>);
  return true;
}
static_assert(check_signed_transformations<signed char, unsigned char>());
static_assert(check_signed_transformations<short, unsigned short>());
static_assert(check_signed_transformations<int, unsigned>());
static_assert(check_signed_transformations<long, unsigned long>());
static_assert(check_signed_transformations<long long, unsigned long long>());
static_assert(check_signed_transformations<__int128, unsigned __int128>());
enum class enumeration : unsigned {};
static_assert(traits::is_same_v<traits::make_signed_t<enumeration const>, int const>);
static_assert(traits::is_same_v<traits::make_unsigned_t<enumeration volatile>, unsigned volatile>);

static_assert(traits::integral_constant<int, 17>::value == 17);
static_assert(traits::integral_constant<int, 17>{}() == 17);
static_assert(int(traits::integral_constant<int, 17>{}) == 17);
static_assert(traits::is_same_v<typename traits::integral_constant<int, 17>::type,
                                traits::integral_constant<int, 17>>);
static_assert(traits::is_same_v<traits::conditional_t<true, int, incomplete>, int>);
static_assert(
  traits::is_same_v<typename traits::conditional<false, incomplete, double>::type, double>);
static_assert(traits::is_same_v<traits::enable_if_t<true>, void>);
static_assert(traits::is_same_v<typename traits::enable_if<true, int>::type, int>);
template <bool Condition>
concept enabled = requires { typename traits::enable_if<Condition>::type; };
static_assert(enabled<true> && !enabled<false>);
static_assert(traits::is_same_v<traits::void_t<int, void, incomplete>, void>);

struct no_value;
static_assert(!traits::disjunction_v<>);
static_assert(!traits::disjunction_v<traits::false_type, traits::false_type>);
static_assert(traits::disjunction_v<traits::false_type, traits::true_type, no_value>);
static_assert(traits::disjunction<traits::integral_constant<int, 3>, no_value>::value == 3);

struct base {};
struct public_derived : base {};
struct private_derived : private base {};
struct unrelated {};
CHECK_TRAIT(is_base_of, true, base, base);
CHECK_TRAIT(is_base_of, true, base const, public_derived volatile);
CHECK_TRAIT(is_base_of, true, base, private_derived);
CHECK_TRAIT(is_base_of, false, public_derived, base);
CHECK_TRAIT(is_base_of, false, base, unrelated);
CHECK_TRAIT(is_base_of, false, int, int);
CHECK_TRAIT(is_convertible, true, public_derived*, base*);
CHECK_TRAIT(is_convertible, false, private_derived*, base*);
CHECK_TRAIT(is_convertible, false, base*, public_derived*);
