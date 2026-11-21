/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "test_support.cuh"

#include <cuda/std/concepts>
#include <cuda/std/type_traits>

template <class T, bool Default, bool CopyConstruct, bool MoveConstruct, bool Move, bool Copy>
constexpr bool check_object()
{
  CHECK_CONCEPT(destructible, true, T);
  CHECK_CONCEPT(constructible_from, traits::is_default_constructible_v<T>, T);
  CHECK_CONCEPT(default_initializable, Default, T);
  CHECK_CONCEPT(copy_constructible, CopyConstruct, T);
  CHECK_CONCEPT(move_constructible, MoveConstruct, T);
  CHECK_CONCEPT(movable, Move, T);
  CHECK_CONCEPT(copyable, Copy, T);
  return true;
}

enum class enumeration : unsigned {};
using scalars = test_types<bool,
                           char,
                           short,
                           int,
                           unsigned,
                           long,
                           long long,
                           __int128,
                           unsigned __int128,
                           float,
                           double,
                           int*,
                           decltype(nullptr),
                           enumeration>;
static_assert(for_types(scalars{}, []<class T> {
  static_assert(check_object<T, true, true, true, true, true>());
  static_assert(check_object<T volatile, true, true, true, true, true>());
  static_assert(check_object<T const, false, true, true, false, false>());
  static_assert(check_object<T const volatile, false, true, true, false, false>());
  static_assert(check_object<T&, false, true, true, false, false>());
  static_assert(check_object<T&&, false, false, true, false, false>());
  static_assert(check_object<T const&, false, true, true, false, false>());
  static_assert(check_object<T const&&, false, false, true, false, false>());
  static_assert(check_object<T[2], true, false, false, false, false>());
  static_assert(check_object<T const[2], false, false, false, false, false>());
  CHECK_TRAIT(is_assignable, true, T&, T);
  CHECK_TRAIT(is_assignable, true, T&, T const&);
  CHECK_TRAIT(is_assignable, true, T volatile&, T);
  CHECK_TRAIT(is_assignable, false, T, T);
  CHECK_TRAIT(is_assignable, false, T const&, T);
  CHECK_TRAIT(is_assignable, false, T const volatile&, T);
  return true;
}));

using function = void();
static_assert(check_object<function&, false, true, true, false, false>());
static_assert(check_object<int[2][3], true, false, false, false, false>());

struct copy_only {
  copy_only() = default;
  copy_only(copy_only const&) {}
  copy_only& operator=(copy_only const&) { return *this; }
};
// Moving can use copying when no move operation is declared.
static_assert(check_object<copy_only, true, true, true, true, true>());

struct explicit_default {
  explicit explicit_default() {}
};
static_assert(check_object<explicit_default, true, true, true, true, true>());

struct reference_member {
  int& reference;
};
static_assert(check_object<reference_member, false, true, true, false, false>());
struct const_member {
  int const value;
};
static_assert(check_object<const_member, false, true, true, false, false>());
