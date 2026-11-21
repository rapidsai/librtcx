/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "test_support.cuh"

#include <cuda/std/concepts>
#include <cuda/std/type_traits>

template <class T>
constexpr int arithmetic_rank(T)
{
  return 0;
}
template <traits::integral T>
constexpr int arithmetic_rank(T)
{
  return 1;
}
template <traits::signed_integral T>
constexpr int arithmetic_rank(T)
{
  return 2;
}
template <traits::unsigned_integral T>
constexpr int arithmetic_rank(T)
{
  return 3;
}
template <traits::floating_point T>
constexpr int arithmetic_rank(T)
{
  return 4;
}

static_assert(arithmetic_rank(-1) == 2);
static_assert(arithmetic_rank(1u) == 3);
static_assert(arithmetic_rank(true) == 3);
static_assert(arithmetic_rank(__int128(-1)) == 2);
static_assert(arithmetic_rank(static_cast<unsigned __int128>(1)) == 3);
static_assert(arithmetic_rank(1.0) == 4);
static_assert(arithmetic_rank(nullptr) == 0);

template <class T>
constexpr int construction_rank()
{
  return 0;
}
template <traits::destructible T>
constexpr int construction_rank()
{
  return 1;
}
template <traits::constructible_from T>
constexpr int construction_rank()
{
  return 2;
}
template <traits::default_initializable T>
constexpr int construction_rank()
{
  return 3;
}

struct throwing_destructor {
  ~throwing_destructor() noexcept(false);
};
struct no_default {
  no_default() = delete;
};
static_assert(construction_rank<void>() == 0);
#ifdef LIBNVCXX
static_assert(construction_rank<throwing_destructor>() == 3);
#else
static_assert(construction_rank<throwing_destructor>() == 0);
#endif
static_assert(construction_rank<no_default>() == 1);
static_assert(construction_rank<int const>() == 2);
static_assert(construction_rank<int>() == 3);

template <class T>
constexpr int object_rank()
{
  return 0;
}
template <traits::movable T>
constexpr int object_rank()
{
  return 1;
}
template <traits::copyable T>
constexpr int object_rank()
{
  return 2;
}

struct move_only {
  move_only(move_only&&)            = default;
  move_only& operator=(move_only&&) = default;
};
static_assert(object_rank<int>() == 2);
static_assert(object_rank<move_only>() == 1);
static_assert(object_rank<int const>() == 0);
static_assert(object_rank<int&>() == 0);

// Failure must remove a constrained overload, not cause a hard instantiation error.
template <class LHS, class RHS>
constexpr bool assignment_available()
{
  return false;
}
template <class LHS, class RHS>
  requires traits::is_assignable_v<LHS, RHS>
constexpr bool assignment_available()
{
  return true;
}
static_assert(assignment_available<int&, int>());
static_assert(!assignment_available<void, void>());
static_assert(!assignment_available<int const&, int>());

template <class T, class U>
constexpr bool conversion_available()
{
  return false;
}
template <class T, class U>
  requires traits::convertible_to<T, U>
constexpr bool conversion_available()
{
  return true;
}
static_assert(conversion_available<void, void>());
static_assert(!conversion_available<void, int>());

#ifdef LIBNVCXX
// Regression: direct is_same_v expressions do not provide symmetric subsumption in NVRTC.
template <class T, class U>
  requires traits::same_as<T, U>
constexpr int same_rank()
{
  return 1;
}
template <class T, class U>
  requires traits::same_as<U, T> && traits::integral<T>
constexpr int same_rank()
{
  return 2;
}
static_assert(same_rank<int, int>() == 2);
#endif
