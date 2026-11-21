/* SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

/** @file function_ref.hpp
 * @brief Non-owning callable references adapted from rtcx::func.
 */

#include <memory>
#include <type_traits>
#include <utility>

namespace cr
{
/**
 * @brief Non-owning reference to a callable with the specified signature.
 * @tparam Signature Return type and argument types of the borrowed callable.
 */
template <typename Signature> struct function_ref;

/**
 * @brief Zero-copy type erasure for synchronous callbacks; the callable must outlive its use.
 */
template <typename R, typename... Args> struct [[nodiscard]] function_ref<R(Args...)>
{
  private:
    void *user_data_;

    R (*thunk_)(void *, Args...);

  public:
    /**
     * @brief Borrow user data and an invocation thunk.
     * @param user_data Context that must outlive all invocations.
     * @param thunk Function that invokes the callable represented by user_data.
     */
    function_ref(void *user_data, R (*thunk)(void *, Args...)) : user_data_(user_data), thunk_(thunk)
    {
    }

    /**
     * @brief Reference a function pointer without allocating storage.
     * @param function Non-null callable entry point.
     */
    function_ref(R (*function)(Args...))
        : function_ref(
              reinterpret_cast<void *>(function), +[](void *data, Args... args) -> R {
                  return reinterpret_cast<R (*)(Args...)>(data)(std::forward<Args>(args)...);
              })
    {
    }

    /**
     * @brief Borrow a functor; a temporary remains valid only until the enclosing expression ends.
     * @tparam Lambda Callable type, including reference and const qualification.
     * @param lambda Callable that must outlive all invocations of this reference.
     */
    template <typename Lambda>
        requires(!std::is_same_v<std::remove_cvref_t<Lambda>, function_ref> &&
                 std::is_object_v<std::remove_reference_t<Lambda>> && std::is_invocable_r_v<R, Lambda &, Args...>)
    function_ref(Lambda &&lambda) : function_ref(from_functor(lambda))
    {
    }

    /**
     * @brief Invoke the borrowed callable with forwarded arguments.
     * @param args Arguments forwarded to the callable.
     * @return The callable's result.
     */
    R operator()(Args... args) const
    {
        return thunk_(user_data_, std::forward<Args>(args)...);
    }

    /**
     * @brief Borrow a functor using rtcx's user-data and thunk representation.
     * @tparam Lambda Borrowed callable type.
     * @param lambda Callable that must outlive all invocations of the returned reference.
     * @return Non-owning callable reference.
     */
    template <typename Lambda> static function_ref from_functor(Lambda &lambda)
    {
        return {
            const_cast<void *>(static_cast<void const *>(std::addressof(lambda))),
            +[](void *data, Args... args) -> R { return (*static_cast<Lambda *>(data))(std::forward<Args>(args)...); }};
    }
};

} // namespace cr
