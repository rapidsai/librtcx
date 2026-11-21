/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "test_support.cuh"

#include <cuda/std/type_traits>

template <class T,
          bool Default,
          bool Copy,
          bool Move,
          bool CopyAssign,
          bool MoveAssign,
          bool Destroy,
          bool Trivial,
          bool TrivialCopy>
constexpr bool check_operations()
{
  CHECK_TRAIT(is_default_constructible, Default, T);
  CHECK_TRAIT(is_copy_constructible, Copy, T);
  CHECK_TRAIT(is_move_constructible, Move, T);
  CHECK_TRAIT(is_copy_assignable, CopyAssign, T);
  CHECK_TRAIT(is_move_assignable, MoveAssign, T);
  CHECK_TRAIT(is_destructible, Destroy, T);
  CHECK_TRAIT(is_trivial, Trivial, T);
  CHECK_TRAIT(is_trivially_copyable, TrivialCopy, T);
  return true;
}

using scalars = test_types<bool,
                           char,
                           signed char,
                           unsigned char,
                           wchar_t,
                           char8_t,
                           char16_t,
                           char32_t,
                           short,
                           unsigned short,
                           int,
                           unsigned,
                           long,
                           unsigned long,
                           long long,
                           unsigned long long,
                           __int128,
                           unsigned __int128,
                           float,
                           double,
                           long double,
                           int*,
                           decltype(nullptr)>;
static_assert(for_types(scalars{}, []<class T> {
  static_assert(check_operations<T, true, true, true, true, true, true, true, true>());
  static_assert(check_operations<T const, true, true, true, false, false, true, true, true>());
  static_assert(check_operations<T volatile, true, true, true, true, true, true, true, true>());
  static_assert(
    check_operations<T const volatile, true, true, true, false, false, true, true, true>());
  static_assert(check_operations<T&, false, true, true, true, true, true, false, false>());
  static_assert(check_operations<T const&, false, true, true, false, false, true, false, false>());
  static_assert(check_operations<T&&, false, false, true, true, true, true, false, false>());
  static_assert(
    check_operations<T const&&, false, false, true, false, false, true, false, false>());
  return true;
}));

struct aggregate {
  int value;
};
struct user_default {
  user_default() {}
};
struct copy_only {
  copy_only() = default;
  copy_only(copy_only const&) {}
  copy_only& operator=(copy_only const&) { return *this; }
};
struct move_only {
  move_only() = default;
  move_only(move_only&&) {}
  move_only& operator=(move_only&&) { return *this; }
};
struct deleted_move {
  deleted_move()                               = default;
  deleted_move(deleted_move const&)            = default;
  deleted_move(deleted_move&&)                 = delete;
  deleted_move& operator=(deleted_move const&) = default;
  deleted_move& operator=(deleted_move&&)      = delete;
};
struct virtual_member {
  virtual void function();
};
struct deleted_default {
  deleted_default() = delete;
};
struct throwing_destructor {
  ~throwing_destructor() noexcept(false) {}
};
struct private_destructor {
 private:
  ~private_destructor() = default;
};

static_assert(check_operations<aggregate, true, true, true, true, true, true, true, true>());
static_assert(check_operations<user_default, true, true, true, true, true, true, false, true>());
static_assert(check_operations<copy_only, true, true, true, true, true, true, false, false>());
static_assert(check_operations<move_only, true, false, true, false, true, true, false, false>());
static_assert(check_operations<deleted_move, true, true, false, true, false, true, true, true>());
static_assert(check_operations<virtual_member, true, true, true, true, true, true, false, false>());
static_assert(
  check_operations<deleted_default, false, true, true, true, true, true, false, true>());
static_assert(
  check_operations<throwing_destructor, true, true, true, true, true, true, false, false>());
static_assert(check_operations<int[2][3], true, false, false, false, false, true, true, true>());
static_assert(check_operations<int[], false, false, false, false, false, false, true, true>());
static_assert(
  check_operations<void const volatile, false, false, false, false, false, false, false, false>());
static_assert(check_operations<void(), false, false, false, false, false, false, false, false>());
// References do not destroy their referents, even when the destructor is inaccessible.
CHECK_TRAIT(is_destructible, false, private_destructor);
CHECK_TRAIT(is_destructible, true, private_destructor&);
CHECK_TRAIT(is_destructible, true, throwing_destructor&&);
CHECK_TRAIT(is_destructible, true, throwing_destructor[2]);

struct constructor_overloads {
  constructor_overloads() = delete;
  explicit constructor_overloads(int);
  constructor_overloads(int, double);
  constructor_overloads(char const*) = delete;
};
struct private_constructor {
 private:
  private_constructor(int);
};
struct abstract {
  virtual void function() = 0;
};
struct base {};
struct public_derived : base {};
struct private_derived : private base {};
struct left : base {};
struct right : base {};
struct ambiguous_derived : left, right {};

CHECK_TRAIT(is_constructible, true, constructor_overloads, int);
CHECK_TRAIT(is_constructible, true, constructor_overloads, short, float);
CHECK_TRAIT(is_constructible, false, constructor_overloads);
CHECK_TRAIT(is_constructible, false, constructor_overloads, char const*);
CHECK_TRAIT(is_constructible, false, constructor_overloads, int, int, int);
CHECK_TRAIT(is_constructible, false, private_constructor, int);
CHECK_TRAIT(is_constructible, false, abstract);
CHECK_TRAIT(is_constructible, true, base&, public_derived&);
CHECK_TRAIT(is_constructible, false, base&, private_derived&);
CHECK_TRAIT(is_constructible, false, base&, ambiguous_derived&);
CHECK_TRAIT(is_constructible, false, public_derived&, base&);
CHECK_TRAIT(is_constructible, true, int const&, int);
CHECK_TRAIT(is_constructible, false, int&&, int&);
CHECK_TRAIT(is_constructible, false, int&, int const&);

struct ref_qualified_assignment {
  ref_qualified_assignment& operator=(int) &;
  void operator=(double) &&;
};
CHECK_TRAIT(is_assignable, true, ref_qualified_assignment&, int);
CHECK_TRAIT(is_assignable, true, ref_qualified_assignment&&, double);
CHECK_TRAIT(is_assignable, true, ref_qualified_assignment, double);
CHECK_TRAIT(is_assignable, false, ref_qualified_assignment const&, int);
CHECK_TRAIT(is_assignable, false, void, void);
CHECK_TRAIT(is_assignable, false, int (&)[2], int (&)[2]);
CHECK_TRAIT(is_assignable, true, int volatile&, int);
