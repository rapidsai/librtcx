/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* SPDX-License-Identifier: Apache-2.0 */

/**
 * @file error_macros_test.cpp
 * @brief Test failure-macro evaluation, exception construction, and source-location reporting.
 *
 * Covers success and failure paths, custom exception contexts, move-only ownership, and if/else composition.
 */

#include "error.hpp"

#include <gtest/gtest.h>
#include <memory>
#include <string_view>
#include <typeinfo>

namespace
{
void expect_location(std::exception const &error, int line, std::string_view expected)
{
    auto message = std::string_view(error.what());
    EXPECT_TRUE(message.ends_with(std::format(": {}", expected))) << message;
    EXPECT_TRUE(message.starts_with("clangrtc failure at ")) << message;
    EXPECT_NE(message.find(std::format("error_macros_test.cpp:{}: ", line)), std::string_view::npos) << message;
}

struct ContextError : std::runtime_error
{
    int code;

    ContextError(int code, std::string message) : std::runtime_error(std::move(message)), code(code)
    {
    }
};

struct OwnedContextError : std::runtime_error
{
    std::unique_ptr<int> context;

    OwnedContextError(std::unique_ptr<int> context, std::string message)
        : std::runtime_error(std::move(message)), context(std::move(context))
    {
    }
};

} // namespace

/**
 * @brief Avoid evaluating diagnostic and context expressions when a precondition succeeds.
 *
 * Counts condition, message, and context evaluations to verify that only the condition is evaluated.
 */
TEST(ErrorMacros, SuccessfulPreconditionDoesNotEvaluateFailureArguments)
{
    int conditions = 0, messages = 0, contexts = 0;
    CR_EXPECTS(++conditions == 1, (++messages, "must not evaluate"), ContextError, ++contexts);
    EXPECT_EQ(conditions, 1);
    EXPECT_EQ(messages, 0);
    EXPECT_EQ(contexts, 0);
}

/**
 * @brief Evaluate failed-precondition arguments once and preserve the macro call location.
 *
 * Checks the default failure status, message suffix, evaluation counters, and expected source line.
 */
TEST(ErrorMacros, FailedPreconditionEvaluatesOnceAndReportsCallsite)
{
    int conditions = 0, messages = 0;
    auto line = __LINE__ + 4;

    try
    {
        CR_EXPECTS(++conditions == 0, (++messages, "constructed diagnostic"));
        FAIL() << "Precondition did not throw";
    }
    catch (cr::Failure const &error)
    {
        EXPECT_EQ(error.status, CLANGRTC_INVALID_ARGUMENT);
        expect_location(error, line, "constructed diagnostic");
    }

    EXPECT_EQ(conditions, 1);
    EXPECT_EQ(messages, 1);
}

/**
 * @brief Construct the caller-selected exception type for a failed precondition.
 *
 * Catches the requested standard exception and checks its dynamic type and diagnostic source location.
 */
TEST(ErrorMacros, PreconditionUsesRequestedExceptionType)
{
    auto line = __LINE__ + 4;

    try
    {
        CR_EXPECTS(false, "typed precondition", std::runtime_error);
        FAIL() << "Precondition did not throw";
    }
    catch (std::runtime_error const &error)
    {
        EXPECT_EQ(typeid(error), typeid(std::runtime_error));
        expect_location(error, line, "typed precondition");
    }
}

/**
 * @brief Pass a failed-precondition context to the exception constructor exactly once.
 *
 * Uses counters and a context-bearing exception to check value forwarding and the resulting call-site diagnostic.
 */
TEST(ErrorMacros, PreconditionPassesContextOnce)
{
    int conditions = 0, messages = 0, contexts = 0;
    auto line = __LINE__ + 4;

    try
    {
        CR_EXPECTS(++conditions == 0, (++messages, "context diagnostic"), ContextError, (++contexts, 73));
        FAIL() << "Precondition did not throw";
    }
    catch (ContextError const &error)
    {
        EXPECT_EQ(error.code, 73);
        expect_location(error, line, "context diagnostic");
    }

    EXPECT_EQ(conditions, 1);
    EXPECT_EQ(messages, 1);
    EXPECT_EQ(contexts, 1);
}

/**
 * @brief Evaluate the default failure message only once.
 *
 * Checks the resulting clangrtc failure category and source location after an unconditional failure.
 */
TEST(ErrorMacros, DefaultFailureEvaluatesMessageOnce)
{
    int messages = 0;
    auto line = __LINE__ + 4;

    try
    {
        CR_FAIL((++messages, "unconditional failure"));
    }
    catch (cr::Failure const &error)
    {
        EXPECT_EQ(error.status, CLANGRTC_INVALID_ARGUMENT);
        expect_location(error, line, "unconditional failure");
        EXPECT_EQ(messages, 1);
        return;
    }

    FAIL() << "CR_FAIL did not throw";
}

/**
 * @brief Honor the explicitly selected exception type for an unconditional failure.
 *
 * Catches the requested standard exception and verifies its dynamic type and source-location diagnostic.
 */
TEST(ErrorMacros, FailureUsesRequestedExceptionType)
{
    auto line = __LINE__ + 4;

    try
    {
        CR_FAIL("typed unconditional failure", std::runtime_error);
    }
    catch (std::runtime_error const &error)
    {
        EXPECT_EQ(typeid(error), typeid(std::runtime_error));
        expect_location(error, line, "typed unconditional failure");
        return;
    }

    FAIL() << "CR_FAIL did not throw";
}

/**
 * @brief Forward an unconditional failure context without repeating side effects.
 *
 * Checks context and message evaluation counts and the values stored by a custom exception.
 */
TEST(ErrorMacros, FailurePassesContextOnce)
{
    int messages = 0, contexts = 0;
    auto line = __LINE__ + 4;

    try
    {
        CR_FAIL((++messages, "context diagnostic"), ContextError, (++contexts, 73));
    }
    catch (ContextError const &error)
    {
        EXPECT_EQ(error.code, 73);
        expect_location(error, line, "context diagnostic");
        EXPECT_EQ(messages, 1);
        EXPECT_EQ(contexts, 1);
        return;
    }

    FAIL() << "CR_FAIL did not throw";
}

/**
 * @brief Transfer move-only context ownership into a failure exception.
 *
 * Moves a unique pointer through the macro and verifies both the stored payload and the emptied source owner.
 */
TEST(ErrorMacros, FailureTransfersMoveOnlyContext)
{
    auto owned = std::make_unique<int>(91);
    auto line = __LINE__ + 4;

    try
    {
        CR_FAIL("owned context", OwnedContextError, std::move(owned));
    }
    catch (OwnedContextError const &error)
    {
        ASSERT_NE(error.context, nullptr);
        EXPECT_EQ(*error.context, 91);
        EXPECT_EQ(owned, nullptr);
        expect_location(error, line, "owned context");
        return;
    }

    FAIL() << "CR_FAIL did not throw";
}

/**
 * @brief Compose the precondition macro safely inside an unbraced if/else statement.
 *
 * Checks that the surrounding else branch belongs to the caller rather than the macro expansion.
 */
TEST(ErrorMacros, PreconditionDoesNotCaptureOuterElse)
{
    // Deliberately unbraced: the macro must behave like one statement.
    // clang-format off
    int branches = 0;

    if (true)
        CR_EXPECTS(true, "must not throw");
    else
        ++branches;
    // clang-format on
    EXPECT_EQ(branches, 0);
}
