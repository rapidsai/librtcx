/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "test_support.cuh"

#include <cuda/std/concepts>
#include <cuda/std/type_traits>

enum category {
  other_type          = 0,
  integer_type        = 1,
  floating_type       = 2,
  void_type           = 4,
  null_type           = 8,
  pointer_type        = 16,
  member_pointer_type = 32,
  enum_type           = 64,
  function_type       = 128,
  lvalue_type         = 256,
  rvalue_type         = 512,
  signed_type         = 1024,
  unsigned_type       = 2048,
};

template <class T, unsigned Categories, bool Const = false, bool Volatile = false>
constexpr bool check_classification()
{
  constexpr bool integral    = Categories & integer_type;
  constexpr bool floating    = Categories & floating_type;
  constexpr bool arithmetic  = integral || floating;
  constexpr bool fundamental = arithmetic || (Categories & (void_type | null_type));
  constexpr bool reference   = Categories & (lvalue_type | rvalue_type);
  constexpr bool function    = Categories & function_type;
  constexpr bool scalar =
    arithmetic || (Categories & (null_type | pointer_type | member_pointer_type | enum_type));
  CHECK_TRAIT(is_integral, integral, T);
  CHECK_TRAIT(is_floating_point, floating, T);
  CHECK_TRAIT(is_arithmetic, arithmetic, T);
  CHECK_TRAIT(is_fundamental, fundamental, T);
  CHECK_TRAIT(is_compound, !fundamental, T);
  CHECK_TRAIT(is_scalar, scalar, T);
  CHECK_TRAIT(is_void, bool(Categories & void_type), T);
  CHECK_TRAIT(is_pointer, bool(Categories & pointer_type), T);
  CHECK_TRAIT(is_enum, bool(Categories & enum_type), T);
  CHECK_TRAIT(is_function, function, T);
  CHECK_TRAIT(is_object, !(Categories & void_type) && !reference && !function, T);
  CHECK_TRAIT(is_lvalue_reference, bool(Categories & lvalue_type), T);
  CHECK_TRAIT(is_signed, bool(Categories & signed_type), T);
  CHECK_TRAIT(is_unsigned, bool(Categories & unsigned_type), T);
  CHECK_TRAIT(is_const, Const && !reference && !function, T);
  CHECK_TRAIT(is_volatile, Volatile && !reference && !function, T);
  CHECK_CONCEPT(integral, integral, T);
  CHECK_CONCEPT(signed_integral, integral && bool(Categories & signed_type), T);
  CHECK_CONCEPT(unsigned_integral, integral && bool(Categories & unsigned_type), T);
  CHECK_CONCEPT(floating_point, floating, T);
  return true;
}

template <class T, unsigned Categories>
constexpr bool check_family()
{
  static_assert(check_classification<T, Categories>());
  static_assert(check_classification<T const, Categories, true>());
  static_assert(check_classification<T volatile, Categories, false, true>());
  static_assert(check_classification<T const volatile, Categories, true, true>());
  if constexpr (requires { typename traits::void_t<T&>; }) {
    static_assert(check_classification<T&, lvalue_type>());
    static_assert(check_classification<T const&, lvalue_type>());
    static_assert(check_classification<T volatile&, lvalue_type>());
    static_assert(check_classification<T const volatile&, lvalue_type>());
    static_assert(check_classification<T&&, rvalue_type>());
    static_assert(check_classification<T const&&, rvalue_type>());
    static_assert(check_classification<T volatile&&, rvalue_type>());
    static_assert(check_classification<T const volatile&&, rvalue_type>());
  }
  return true;
}

using signed_integers   = test_types<signed char, short, int, long, long long, __int128>;
using unsigned_integers = test_types<bool,
                                     unsigned char,
                                     char8_t,
                                     char16_t,
                                     char32_t,
                                     unsigned short,
                                     unsigned int,
                                     unsigned long,
                                     unsigned long long,
                                     unsigned __int128>;
static_assert(for_types(signed_integers{},
                        []<class T> { return check_family<T, integer_type | signed_type>(); }));
static_assert(for_types(unsigned_integers{},
                        []<class T> { return check_family<T, integer_type | unsigned_type>(); }));
static_assert(check_family<char, integer_type | (char(-1) < 0 ? signed_type : unsigned_type)>());
static_assert(
  check_family<wchar_t, integer_type | (wchar_t(-1) < 0 ? signed_type : unsigned_type)>());
static_assert(for_types(test_types<float, double, long double>{},
                        []<class T> { return check_family<T, floating_type | signed_type>(); }));

struct incomplete;
struct empty {};
struct record {
  int member;
  void function() const;
};
union storage {
  int integer;
  double floating;
};
enum unscoped : unsigned { enumerator };
enum class scoped : int {};
using function           = int(double, ...);
using qualified_function = int() const volatile&& noexcept;

static_assert(check_family<void, void_type>());
static_assert(check_family<decltype(nullptr), null_type>());
static_assert(for_types(test_types<int*, int const*, void*, incomplete*, function*>{},
                        []<class T> { return check_family<T, pointer_type>(); }));
static_assert(check_family<int record::*, member_pointer_type>());
static_assert(check_family<void (record::*)() const, member_pointer_type>());
static_assert(check_family<unscoped, enum_type>());
static_assert(check_family<scoped, enum_type>());
static_assert(check_family<function, function_type>());
static_assert(check_family<qualified_function, function_type>());
static_assert(for_types(test_types<incomplete, empty, record, storage, int[3], int[], int[2][3]>{},
                        []<class T> { return check_family<T, other_type>(); }));

// Conversion operators do not make a class an arithmetic or scalar type.
struct integer_like {
  operator int() const;
};
static_assert(check_family<integer_like, other_type>());

// Traits retain the integral_constant interface, in addition to ::value and _v.
static_assert(traits::is_integral<int>{}());
static_assert(!bool(traits::is_integral<int&>{}));
static_assert(traits::is_same_v<typename traits::is_scalar<int*>::value_type, bool>);
static_assert(traits::is_same_v<typename traits::is_integral<int>::type, traits::true_type>);
