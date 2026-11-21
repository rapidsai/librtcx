/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "test_support.cuh"

#include <cuda/std/type_traits>

enum class enumeration : unsigned {};
struct incomplete;
struct plain {
  int member;
  void function() const;
};
struct nontrivial {
  nontrivial() {}
};
struct noncopyable {
  noncopyable()                              = default;
  noncopyable(noncopyable const&)            = delete;
  noncopyable& operator=(noncopyable const&) = delete;
};
struct movable {
  movable(movable&&)            = default;
  movable& operator=(movable&&) = default;
};
struct explicit_constructor {
  explicit explicit_constructor(int);
};
struct private_constructor {
 private:
  private_constructor();
};
struct deleted_destructor {
  ~deleted_destructor() = delete;
};
using function           = void();
using qualified_function = void() const;

CHECK_TRAIT(is_fundamental, true, void const volatile);
CHECK_TRAIT(is_fundamental, true, decltype(nullptr));
CHECK_TRAIT(is_fundamental, true, unsigned __int128 const);
CHECK_TRAIT(is_fundamental, true, double);
CHECK_TRAIT(is_fundamental, false, enumeration);
CHECK_TRAIT(is_fundamental, false, int*);
CHECK_TRAIT(is_fundamental, false, incomplete);
CHECK_TRAIT(is_fundamental, false, int&);
CHECK_TRAIT(is_lvalue_reference, true, int const&);
CHECK_TRAIT(is_lvalue_reference, false, int&&);
CHECK_TRAIT(is_function, true, qualified_function);
CHECK_TRAIT(is_function, false, function*);
CHECK_TRAIT(is_object, true, int[2]);
CHECK_TRAIT(is_object, true, incomplete);
CHECK_TRAIT(is_object, false, qualified_function);
CHECK_TRAIT(is_object, false, int&);
CHECK_TRAIT(is_object, false, void const);
CHECK_TRAIT(is_arithmetic, true, bool);
CHECK_TRAIT(is_arithmetic, true, __int128);
CHECK_TRAIT(is_arithmetic, true, float const volatile);
CHECK_TRAIT(is_arithmetic, false, enumeration);
CHECK_TRAIT(is_arithmetic, false, void);
CHECK_TRAIT(is_scalar, true, decltype(nullptr) const);
CHECK_TRAIT(is_scalar, true, double);
CHECK_TRAIT(is_scalar, true, enumeration);
CHECK_TRAIT(is_scalar, true, int* const);
CHECK_TRAIT(is_scalar, true, function*);
CHECK_TRAIT(is_scalar, true, int plain::* const volatile);
CHECK_TRAIT(is_scalar, true, void (plain::*)() const);
CHECK_TRAIT(is_scalar, false, void);
CHECK_TRAIT(is_scalar, false, int&);
CHECK_TRAIT(is_scalar, false, int[2]);
CHECK_TRAIT(is_scalar, false, function);
CHECK_TRAIT(is_scalar, false, incomplete);
CHECK_TRAIT(is_compound, false, void);
CHECK_TRAIT(is_compound, false, decltype(nullptr));
CHECK_TRAIT(is_compound, false, int);
CHECK_TRAIT(is_compound, true, enumeration);
CHECK_TRAIT(is_compound, true, int*);
CHECK_TRAIT(is_compound, true, int&);
CHECK_TRAIT(is_compound, true, int[2]);
CHECK_TRAIT(is_compound, true, function);
CHECK_TRAIT(is_compound, true, incomplete);
CHECK_TRAIT(is_const, true, void const);
CHECK_TRAIT(is_const, true, int const volatile);
CHECK_TRAIT(is_const, true, int const[2]);
CHECK_TRAIT(is_const, true, int* const);
CHECK_TRAIT(is_const, false, int const*);
CHECK_TRAIT(is_const, false, int const&);
CHECK_TRAIT(is_const, false, qualified_function);
CHECK_TRAIT(is_volatile, true, void volatile);
CHECK_TRAIT(is_volatile, true, int const volatile);
CHECK_TRAIT(is_volatile, true, int volatile[2]);
CHECK_TRAIT(is_volatile, false, int volatile*);
CHECK_TRAIT(is_volatile, false, int volatile&);
CHECK_TRAIT(is_volatile, false, qualified_function);
CHECK_TRAIT(is_trivial, true, plain);
CHECK_TRAIT(is_trivial, true, int const[2]);
CHECK_TRAIT(is_trivial, false, nontrivial);
CHECK_TRAIT(is_trivial, false, void);
CHECK_TRAIT(is_trivially_copyable, true, nontrivial);
CHECK_TRAIT(is_trivially_copyable, true, plain);
CHECK_TRAIT(is_trivially_copyable, true, int[2]);
CHECK_TRAIT(is_trivially_copyable, false, int&);
CHECK_TRAIT(is_trivially_copyable, false, void);
CHECK_TRAIT(is_destructible, true, int[2]);
CHECK_TRAIT(is_destructible, true, int&);
CHECK_TRAIT(is_destructible, false, int[]);
CHECK_TRAIT(is_destructible, false, deleted_destructor);
CHECK_TRAIT(is_constructible, true, explicit_constructor, int);
CHECK_TRAIT(is_constructible, true, int&, int&);
CHECK_TRAIT(is_constructible, false, int&, int);
CHECK_TRAIT(is_default_constructible, true, int);
CHECK_TRAIT(is_default_constructible, true, int[2]);
CHECK_TRAIT(is_default_constructible, false, int[]);
CHECK_TRAIT(is_default_constructible, false, int&);
CHECK_TRAIT(is_default_constructible, false, private_constructor);
CHECK_TRAIT(is_default_constructible, false, deleted_destructor);
CHECK_TRAIT(is_default_constructible, false, void const volatile);
CHECK_TRAIT(is_copy_constructible, true, int const);
CHECK_TRAIT(is_copy_constructible, true, int&);
CHECK_TRAIT(is_copy_constructible, false, int&&);
CHECK_TRAIT(is_copy_constructible, false, int[2]);
CHECK_TRAIT(is_copy_constructible, false, noncopyable);
CHECK_TRAIT(is_copy_constructible, false, movable);
CHECK_TRAIT(is_copy_constructible, false, void const volatile);
CHECK_TRAIT(is_copy_constructible, false, function);
CHECK_TRAIT(is_copy_constructible, false, qualified_function);
CHECK_TRAIT(is_move_constructible, true, movable);
CHECK_TRAIT(is_move_constructible, true, int&&);
CHECK_TRAIT(is_move_constructible, false, noncopyable);
CHECK_TRAIT(is_move_constructible, false, int[2]);
CHECK_TRAIT(is_move_constructible, false, void const volatile);
CHECK_TRAIT(is_move_constructible, false, qualified_function);
CHECK_TRAIT(is_assignable, true, int&, double);
CHECK_TRAIT(is_assignable, false, int, int);
CHECK_TRAIT(is_assignable, false, int const&, int);
CHECK_TRAIT(is_copy_assignable, true, int);
CHECK_TRAIT(is_copy_assignable, true, int&);
CHECK_TRAIT(is_copy_assignable, false, int const);
CHECK_TRAIT(is_copy_assignable, false, int[2]);
CHECK_TRAIT(is_copy_assignable, false, noncopyable);
CHECK_TRAIT(is_copy_assignable, false, movable);
CHECK_TRAIT(is_copy_assignable, false, void const volatile);
CHECK_TRAIT(is_copy_assignable, false, qualified_function);
CHECK_TRAIT(is_move_assignable, true, movable);
CHECK_TRAIT(is_move_assignable, true, int&&);
CHECK_TRAIT(is_move_assignable, false, noncopyable);
CHECK_TRAIT(is_move_assignable, false, int const);
CHECK_TRAIT(is_move_assignable, false, int[2]);
CHECK_TRAIT(is_move_assignable, false, void const volatile);
CHECK_TRAIT(is_move_assignable, false, qualified_function);

static_assert(traits::is_constant_evaluated());
static_assert(noexcept(traits::is_constant_evaluated()));
constexpr int evaluation_context() { return traits::is_constant_evaluated() ? 1 : 2; }
static_assert(evaluation_context() == 1);

extern "C" __global__ void runtime_evaluation_context(int* result)
{
  *result = evaluation_context();  // Must produce 2 at runtime.
}

#ifdef NVCXX_AS_STD
static_assert(cuda::std::is_scalar_v<int*>);
static_assert(std::is_move_constructible_v<movable>);
#endif

#undef CHECK_TRAIT
