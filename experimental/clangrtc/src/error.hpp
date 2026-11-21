/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* SPDX-License-Identifier: Apache-2.0 */
#pragma once

/**
 * @file error.hpp
 * @brief Internal status-bearing exceptions and call-site-aware failure macros.
 */

#include "clangrtc.h"

#include <exception>
#include <format>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace cr
{
/**
 * @brief RAII utility to execute a callable at the end of a scope.
 */
template <typename T> struct defer
{
  private:
    T func_; ///< The callable to be executed at the end of the scope

  public:
    /** @brief Construct the callable that will run when this guard leaves scope. */
    template <typename... Args> defer(Args &&...args) : func_{static_cast<Args &&>(args)...}
    {
    }

    /** @brief Disable copying to keep cleanup owned by one guard. */
    defer(defer const &) = delete;

    /** @brief Disable copy assignment to preserve the registered cleanup. */
    defer &operator=(defer const &) = delete;

    /** @brief Disable moving to keep cleanup bound to its original scope. */
    defer(defer &&) = delete;

    /** @brief Disable move assignment to preserve the registered cleanup. */
    defer &operator=(defer &&) = delete;

    /** @brief Invoke the registered cleanup exactly once at scope exit. */
    ~defer()
    {
        func_();
    }
};

/** @brief Deduce the stored callable type from a cleanup expression. */
template <typename T> defer(T) -> defer<T>;

/**
 * @brief Carries an API status together with an internal failure message.
 */
struct Failure : std::runtime_error
{
    /**
     * @brief API status returned when this failure reaches a public entry point.
     */
    clangrtcStatus status;

    /**
     * @brief Producing diagnostic stage.
     */
    std::string stage;

    /**
     * @brief Constructs a status-bearing internal exception.
     *
     * @param status API failure category.
     * @param message Diagnostic text copied into the exception.
     * @param stage Producing diagnostic stage.
     */
    Failure(clangrtcStatus status, std::string message, std::string stage)
        : std::runtime_error(std::move(message)), status(status), stage(std::move(stage))
    {
    }

    /**
     * @brief Constructs a frontend failure with an explicit status.
     * @param status API failure category.
     * @param message Diagnostic text copied into the exception.
     */
    Failure(clangrtcStatus status, std::string message) : Failure(status, std::move(message), "frontend")
    {
    }

    /**
     * @brief Constructs an invalid-argument failure from a diagnostic alone.
     *
     * @param message Diagnostic text copied into the exception.
     */
    explicit Failure(std::string message) : Failure(CLANGRTC_INVALID_ARGUMENT, std::move(message))
    {
    }
};

} // namespace cr

/**
 * @brief Selects a CR_FAIL implementation by argument count.
 *
 * @param first First supplied argument.
 * @param second Second supplied argument or fallback macro.
 * @param third Third supplied argument or fallback macro.
 * @param selected Implementation selected for the supplied argument count.
 * @param ... Remaining fallback macro names.
 */
#define CR_DETAIL_SELECT_FAIL(first, second, third, selected, ...) selected

/**
 * @brief Throws the selected exception with a formatted call-site diagnostic.
 *
 * The optional context is passed before the message in the exception constructor.
 * Each supplied expression is evaluated exactly once.
 *
 * @param exception_type Exception type derived from std::exception.
 * @param message Expression producing the diagnostic text.
 * @param ... Optional context expression passed to the exception constructor.
 */
#define CR_DETAIL_THROW(exception_type, message, ...)                                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        static_assert(std::is_base_of_v<std::exception, exception_type>);                                              \
        throw exception_type(__VA_OPT__((__VA_ARGS__), )                                                               \
                                 std::format("clangrtc failure at {}:{}: {}", __FILE__, __LINE__, (message)));         \
    } while (false)

/**
 * @brief Throws the default status-bearing exception.
 *
 * @param message Expression producing the diagnostic text.
 */
#define CR_DETAIL_FAIL_1(message) CR_DETAIL_THROW(::cr::Failure, message)

/**
 * @brief Throws a message-only exception of the specified type.
 *
 * @param message Expression producing the diagnostic text.
 * @param exception_type Exception type constructed from the formatted message.
 */
#define CR_DETAIL_FAIL_2(message, exception_type) CR_DETAIL_THROW(exception_type, message)

/**
 * @brief Throws an exception with caller-provided context and diagnostic text.
 *
 * @param message Expression producing the diagnostic text.
 * @param exception_type Exception type constructed from context and message.
 * @param context Extra constructor argument, such as an API error status.
 */
#define CR_DETAIL_FAIL_3(message, exception_type, context) CR_DETAIL_THROW(exception_type, message, context)

/**
 * @brief Throws an exception with the failure call site's filename and line.
 *
 * The default exception is cr::Failure with CLANGRTC_INVALID_ARGUMENT status.
 * An explicit exception type must accept a message, or context followed by a
 * message when context is supplied. All argument expressions are evaluated once.
 *
 * @code{.cpp}
 * CR_FAIL("Unexpected compiler state");
 * CR_FAIL("Unsupported code path", std::runtime_error);
 * CR_FAIL("Invalid compiler option", cr::Failure, CLANGRTC_INVALID_OPTION);
 * @endcode
 *
 * @param ... Message, optional exception type, and optional context argument.
 * @throws std::exception The selected exception type, unconditionally.
 */
#define CR_FAIL(...)                                                                                                   \
    CR_DETAIL_SELECT_FAIL(__VA_ARGS__, CR_DETAIL_FAIL_3, CR_DETAIL_FAIL_2, CR_DETAIL_FAIL_1)(__VA_ARGS__)

/**
 * @brief Checks a precondition and throws a call-site-aware exception on failure.
 *
 * The condition is evaluated exactly once. Message and context expressions are
 * evaluated only when it fails. The default exception is cr::Failure with
 * CLANGRTC_INVALID_ARGUMENT status. The do/while form is a single statement.
 *
 * @code{.cpp}
 * CR_EXPECTS(request != nullptr, "NULL request");
 * CR_EXPECTS(ready, "Compiler is not ready", std::runtime_error);
 * CR_EXPECTS(valid, "Invalid option", cr::Failure, CLANGRTC_INVALID_OPTION);
 * @endcode
 *
 * @param condition Condition that must hold.
 * @param ... Message, optional exception type, and optional context argument.
 * @throws std::exception The selected exception type if condition is false.
 */
#define CR_EXPECTS(condition, ...)                                                                                     \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(condition))                                                                                              \
        {                                                                                                              \
            CR_FAIL(__VA_ARGS__);                                                                                      \
        }                                                                                                              \
    } while (false)

/** @cond CLANGRTC_IMPLEMENTATION */
#define CR_DETAIL_CONCATENATE_IMPL(x, y) x##y
#define CR_DETAIL_CONCATENATE(x, y) CR_DETAIL_CONCATENATE_IMPL(x, y)
/** @endcond */

/**
 * @brief Runs a callable when the enclosing scope exits, including during unwinding.
 *
 * @param ... Callable owned by the scope guard; cleanup must not throw.
 */
#define CR_DEFER(...) ::cr::defer CR_DETAIL_CONCATENATE(cr_defer_, __COUNTER__)(__VA_ARGS__)
