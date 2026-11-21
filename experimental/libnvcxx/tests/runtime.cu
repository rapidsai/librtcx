/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "test_support.cuh"

#include <cuda/std/concepts>
#include <cuda/std/type_traits>

constexpr int evaluation_context() { return traits::is_constant_evaluated() ? 1 : 2; }
static_assert(evaluation_context() == 1);
static_assert(traits::is_constant_evaluated());

template <traits::integral T>
constexpr int classify(T)
{
  return 1;
}

template <traits::signed_integral T>
constexpr int classify(T)
{
  return 2;
}

template <traits::unsigned_integral T>
constexpr int classify(T)
{
  return 3;
}

struct move_only {
  int value;
  constexpr explicit move_only(int x) : value(x) {}
  constexpr move_only(move_only&& other) : value(other.value) { other.value = -1; }
  constexpr move_only& operator=(move_only&& other)
  {
    value       = other.value;
    other.value = -1;
    return *this;
  }
};
static_assert(traits::movable<move_only>);
static_assert(!traits::copyable<move_only>);

template <traits::movable T>
constexpr void move_swap(T& a, T& b)
{
  T temporary(static_cast<T&&>(a));
  a = static_cast<T&&>(b);
  b = static_cast<T&&>(temporary);
}

template <class LHS, class RHS>
  requires traits::is_assignable_v<LHS&, RHS>
constexpr LHS& assign(LHS& destination, RHS&& source)
{
  return destination = static_cast<RHS&&>(source);
}

extern "C" __global__ void runtime_checks(int* result)
{
  int failures           = 0;
  constexpr int constant = evaluation_context();
  failures += constant != 1;
  failures += evaluation_context() != 2;
  failures += traits::is_constant_evaluated();
  failures += classify(-1) != 2;
  failures += classify(1u) != 3;
  failures += classify(true) != 3;
  failures += classify(__int128(-1)) != 2;
  failures += classify(static_cast<unsigned __int128>(1)) != 3;
  move_only first(7), second(19);
  move_swap(first, second);
  failures += first.value != 19 || second.value != 7;
  move_only third(static_cast<move_only&&>(first));
  failures += third.value != 19 || first.value != -1;
  auto& reference = assign(first, static_cast<move_only&&>(second));
  failures += &reference != &first || first.value != 7 || second.value != -1;
  int destination = 0;
  failures += &assign(destination, 3.0) != &destination || destination != 3;
  *result = failures;
}
