/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* SPDX-License-Identifier: Apache-2.0 */

/**
 * @file expressions_test.cpp
 * @brief Test expression-slot validation and diagnostic precedence.
 *
 * Distinguishes invalid selectors from source or template-body compilation failures and identifies the failing slot.
 */

#include "support.hpp"

namespace test
{
namespace
{
constexpr std::string_view Source = R"CUDA(
    extern "C" __global__ void kernel()
    {
    }

    __host__ int host_function()
    {
        return 0;
    }

    int host_variable = 0;
    __device__ int declared_function();

    template <class T> __device__ int bad_body(T value)
    {
        return value.missing;
    }
)CUDA";

} // namespace

class InvalidExpressionTest : public CompilationTest, public ::testing::WithParamInterface<char const *>
{
};

/**
 * @brief Identify the failing expression slot while rejecting unsupported selections.
 *
 * Places each invalid parameter after a valid kernel expression and requires the diagnostic to identify slot 1 with no
 * output.
 */
TEST_P(InvalidExpressionTest, DiagnosticIdentifiesSlot)
{
    clangrtcString slots[]{to_view("&kernel"), to_view(GetParam())};
    auto result = compile(request(Source, slots), options);
    EXPECT_EQ(result.status, CLANGRTC_INVALID_EXPRESSION);
    EXPECT_TRUE(result.output().empty());
    EXPECT_NE(result.log().find("compilation-slot-1.cu"), std::string::npos);
}

/** @brief Exercise missing names, nondeclarations, host symbols, and undefined device declarations. */
INSTANTIATE_TEST_SUITE_P(Entries, InvalidExpressionTest,
                         ::testing::Values("&missing", "42", "&host_function", "&host_variable", "&declared_function"));

using ExpressionTest = CompilationTest;

/**
 * @brief Classify an invalid instantiated template body as a compilation failure.
 *
 * Uses a syntactically valid selector whose specialization fails during body instantiation rather than selector
 * parsing.
 */
TEST_F(ExpressionTest, TemplateBodyFailureIsCompilationFailure)
{
    auto slot = to_view("&bad_body<int>");
    auto result = compile(request(Source, {&slot, 1}), options);
    EXPECT_EQ(result.status, CLANGRTC_COMPILATION_FAILED);
    EXPECT_TRUE(result.output().empty());
    EXPECT_FALSE(result.log().empty());
}

/**
 * @brief Report source compilation errors before validating expression slots.
 *
 * Combines an error directive with an invalid selector and checks that the source diagnostic determines the result.
 */
TEST_F(ExpressionTest, SourceErrorTakesPrecedenceOverInvalidExpression)
{
    auto slot = to_view("&missing");
    auto result = compile(request("#error source failure\n", {&slot, 1}), options);
    EXPECT_EQ(result.status, CLANGRTC_COMPILATION_FAILED);
    EXPECT_NE(result.log().find("source failure"), std::string::npos);
}
} // namespace test
