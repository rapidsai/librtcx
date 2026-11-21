/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "test_support.cuh"

#include <cuda/std/concepts>
#include <cuda/std/type_traits>

struct plain {};
struct move_only {
  move_only()                            = default;
  move_only(move_only&&)                 = default;
  move_only& operator=(move_only&&)      = default;
  move_only(move_only const&)            = delete;
  move_only& operator=(move_only const&) = delete;
};
struct immovable {
  immovable(immovable&&)            = delete;
  immovable& operator=(immovable&&) = delete;
};
struct explicit_move {
  explicit explicit_move(explicit_move&&);
  explicit_move& operator=(explicit_move&&);
};
struct explicit_copy {
  explicit_copy(explicit_copy&&);
  explicit explicit_copy(explicit_copy const&);
  explicit_copy& operator=(explicit_copy const&);
};
struct throwing_destructor {
  ~throwing_destructor() noexcept(false);
};
struct deleted_destructor {
  ~deleted_destructor() = delete;
};
struct private_default {
 private:
  private_default();
};
struct const_member {
  int const value;
};
struct private_new {
 private:
  static void* operator new(::size_t);
};
struct assignment_returns_void {
  void operator=(assignment_returns_void const&);
};
struct assignment_only_from_const {
  assignment_only_from_const& operator=(assignment_only_from_const const&);
  assignment_only_from_const& operator=(assignment_only_from_const&) = delete;
};
struct unrelated_source {};
struct unrelated_target {
  unrelated_target& operator=(unrelated_source const&);
};
struct implicit_conversion {
  operator int() const;
};
struct explicit_conversion {
  explicit operator int() const;
};
struct conflicting_conversion {
  operator int() const;
  explicit operator int() = delete;
};
struct base {};
struct derived : base {};

CHECK_CONCEPT(convertible_to, true, void, void);
CHECK_CONCEPT(convertible_to, true, int, double);
CHECK_CONCEPT(convertible_to, true, implicit_conversion, int);
CHECK_CONCEPT(convertible_to, false, explicit_conversion, int);
CHECK_CONCEPT(convertible_to, false, conflicting_conversion, int);
CHECK_CONCEPT(convertible_to, false, void, int);
CHECK_CONCEPT(destructible, true, int&);
CHECK_CONCEPT(destructible, true, int[3]);
CHECK_CONCEPT(destructible, false, void);
#ifdef LIBNVCXX
constexpr bool accepts_throwing_destructor = true;
#else
constexpr bool accepts_throwing_destructor = false;
#endif
CHECK_CONCEPT(destructible, accepts_throwing_destructor, throwing_destructor);
CHECK_CONCEPT(destructible, accepts_throwing_destructor, throwing_destructor[2]);
CHECK_CONCEPT(default_initializable, accepts_throwing_destructor, throwing_destructor);
CHECK_CONCEPT(move_constructible, accepts_throwing_destructor, throwing_destructor);
CHECK_CONCEPT(copy_constructible, accepts_throwing_destructor, throwing_destructor);
CHECK_CONCEPT(movable, accepts_throwing_destructor, throwing_destructor);
CHECK_CONCEPT(copyable, accepts_throwing_destructor, throwing_destructor);
CHECK_CONCEPT(destructible, false, deleted_destructor);
CHECK_CONCEPT(constructible_from, true, int, double);
CHECK_CONCEPT(constructible_from, true, int&, int&);
CHECK_CONCEPT(constructible_from, false, int&, int);
CHECK_CONCEPT(constructible_from, accepts_throwing_destructor, throwing_destructor);
CHECK_CONCEPT(constructible_from, false, deleted_destructor);
CHECK_CONCEPT(default_initializable, true, int);
CHECK_CONCEPT(default_initializable, true, int[3]);
CHECK_CONCEPT(default_initializable, true, private_new);
CHECK_CONCEPT(default_initializable, false, int const);
CHECK_CONCEPT(default_initializable, false, int&);
CHECK_CONCEPT(default_initializable, false, int[]);
CHECK_CONCEPT(default_initializable, false, const_member);
CHECK_CONCEPT(default_initializable, false, private_default);
CHECK_CONCEPT(default_initializable, false, void);
CHECK_CONCEPT(move_constructible, true, move_only);
CHECK_CONCEPT(move_constructible, true, int&);
CHECK_CONCEPT(move_constructible, true, int&&);
CHECK_CONCEPT(move_constructible, false, explicit_move);
CHECK_CONCEPT(move_constructible, false, immovable);
CHECK_CONCEPT(move_constructible, false, void);
CHECK_CONCEPT(copy_constructible, true, plain);
CHECK_CONCEPT(copy_constructible, true, int const);
CHECK_CONCEPT(copy_constructible, true, int&);
CHECK_CONCEPT(copy_constructible, false, int&&);
CHECK_CONCEPT(copy_constructible, false, explicit_copy);
CHECK_CONCEPT(copy_constructible, false, move_only);
CHECK_CONCEPT(copy_constructible, false, void);
CHECK_TRAIT(is_assignable, true, int&, double);
CHECK_TRAIT(is_assignable, true, base&, derived);
CHECK_TRAIT(is_assignable, true, move_only&, move_only);
CHECK_TRAIT(is_assignable, false, int, int);
CHECK_TRAIT(is_assignable, false, int const&, int);
CHECK_TRAIT(is_assignable, false, void, int);
CHECK_TRAIT(is_assignable, false, int&, void);
CHECK_TRAIT(is_assignable, true, assignment_returns_void&, assignment_returns_void);
CHECK_TRAIT(is_assignable, true, unrelated_target&, unrelated_source);
CHECK_CONCEPT(movable, true, int);
CHECK_CONCEPT(movable, true, move_only);
CHECK_CONCEPT(movable, false, int const);
CHECK_CONCEPT(movable, false, int&);
CHECK_CONCEPT(movable, false, int[2]);
CHECK_CONCEPT(movable, false, explicit_move);
#ifdef LIBNVCXX
// Object concepts use assignment traits without constraining the assignment return type.
CHECK_CONCEPT(movable, true, assignment_returns_void);
CHECK_CONCEPT(copyable, true, assignment_returns_void);
#else
CHECK_CONCEPT(movable, false, assignment_returns_void);
CHECK_CONCEPT(copyable, false, assignment_returns_void);
#endif
CHECK_CONCEPT(movable, false, void);
CHECK_CONCEPT(copyable, true, int);
CHECK_CONCEPT(copyable, true, plain);
CHECK_CONCEPT(copyable, false, move_only);
CHECK_CONCEPT(copyable, false, int const);
CHECK_CONCEPT(copyable, false, int&);
CHECK_CONCEPT(copyable, false, explicit_copy);
CHECK_CONCEPT(copyable, false, assignment_only_from_const);
CHECK_CONCEPT(copyable, false, void);

#ifdef LIBNVCXX
// Regression check for symmetric constraint subsumption.
template <class T, class U>
  requires traits::same_as<T, U>
constexpr int symmetric_same_as()
{
  return 1;
}

template <class T, class U>
  requires traits::same_as<U, T> && traits::integral<T>
constexpr int symmetric_same_as()
{
  return 2;
}

static_assert(symmetric_same_as<int, int>() == 2);
#endif

#ifdef NVCXX_AS_STD
static_assert(cuda::std::copyable<int>);
static_assert(std::movable<move_only>);
#endif

#undef CHECK_CONCEPT
