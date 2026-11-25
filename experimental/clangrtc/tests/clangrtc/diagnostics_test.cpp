/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* SPDX-License-Identifier: Apache-2.0 */

/**
 * @file diagnostics_test.cpp
 * @brief Test diagnostic reentrancy and the public compiler metadata contract.
 *
 * Invokes nested compilation and preparation from a callback and checks stable version, capability, and status views.
 */

#include "support.hpp"

namespace test
{
namespace
{
struct DiagnosticCount
{
    unsigned calls = 0;
};

void count_diagnostics(clangrtcDiagnostic const *, void *data)
{
    ++static_cast<DiagnosticCount *>(data)->calls;
}

struct ReentryProbe
{
    std::span<clangrtcString const> options;

    std::span<clangrtcString const> preparationOptions;

    DiagnosticCount nested;

    unsigned calls = 0;

    bool invoked = false;

    bool outputs = false;

    clangrtcStatus compiled = CLANGRTC_INTERNAL_ERROR;

    clangrtcStatus prepared = CLANGRTC_INTERNAL_ERROR;
};

// Callbacks cannot throw or use fatal test assertions across the C boundary.
// Record observations here and assert them after the outer operation returns.
void probe_reentry(clangrtcDiagnostic const *, void *data)
{
    auto &probe = *static_cast<ReentryProbe *>(data);
    ++probe.calls;

    if (probe.invoked)
    {
        return;
    }

    probe.invoked = true;
    auto input = request("#warning nested callback\nextern \"C\" __global__ void nested() {}", {});
    clangrtcResult compiled = nullptr, prepared = nullptr;
    probe.compiled = clangrtcCompile(&input, probe.options.data(), probe.options.size(), count_diagnostics,
                                     &probe.nested, &compiled);
    probe.prepared = clangrtcPrepare(&input, probe.preparationOptions.data(), probe.preparationOptions.size(), nullptr,
                                     nullptr, &prepared);
    clangrtcResultView first{}, second{};
    probe.outputs = clangrtcGetResult(compiled, &first) == CLANGRTC_SUCCESS && first.output.size != 0 &&
                    clangrtcGetResult(prepared, &second) == CLANGRTC_SUCCESS && second.output.size != 0;
    clangrtcDestroyResult(compiled);
    clangrtcDestroyResult(prepared);
}
} // namespace

using DiagnosticsTest = CompilationTest;

/**
 * @brief Keep nested compiler calls independent when invoked from a diagnostic callback.
 *
 * Performs compilation and artifact preparation during an outer call and checks callback counts and result ownership.
 */
TEST_F(DiagnosticsTest, ReentrantCallsHaveIndependentCallbacksAndResults)
{
    auto flags = to_views(options);
    auto preparation = flags;
    preparation.push_back(to_view("--artifact=cuda"));
    ReentryProbe probe{flags, preparation};
    auto input = request("#warning outer callback\nextern \"C\" __global__ void outer() {}", {});
    auto result = compile(input, options, probe_reentry, &probe);
    ASSERT_EQ(result.status, CLANGRTC_SUCCESS) << result.log();
    EXPECT_EQ(result.info().status, result.status);
    EXPECT_TRUE(probe.invoked);
    EXPECT_TRUE(probe.outputs);
    EXPECT_GT(probe.nested.calls, 0);
    EXPECT_EQ(probe.compiled, CLANGRTC_SUCCESS);
    EXPECT_EQ(probe.prepared, CLANGRTC_SUCCESS);
    auto calls = probe.calls;
    auto later = compile(input, options);
    EXPECT_EQ(later.status, CLANGRTC_SUCCESS) << later.log();
    EXPECT_EQ(probe.calls, calls);
}

/**
 * @brief Reject a NULL compile request without publishing a result handle.
 *
 * Calls the public C entry point directly and checks invalid-argument status and a NULL output.
 */
TEST(ApiMetadata, RejectsNullRequest)
{
    clangrtcResult result = nullptr;
    EXPECT_EQ(clangrtcCompile(nullptr, nullptr, 0, nullptr, nullptr, &result), CLANGRTC_INVALID_ARGUMENT);
    EXPECT_EQ(result, nullptr);
}

/**
 * @brief Return stable compiler versions and capability views across metadata queries.
 *
 * Checks NULL-output rejection, shared capability storage, and advertised LTO/PTX output support.
 */
TEST(ApiMetadata, CompilerInformationHasStaticStorage)
{
    clangrtcCompilerInfo first{}, second{};
    EXPECT_EQ(clangrtcGetInfo(nullptr), CLANGRTC_INVALID_ARGUMENT);
    ASSERT_EQ(clangrtcGetInfo(&first), CLANGRTC_SUCCESS);
    ASSERT_EQ(clangrtcGetInfo(&second), CLANGRTC_SUCCESS);
    EXPECT_GT(first.capabilityCount, 0);
    EXPECT_EQ(first.capabilities, second.capabilities);
    EXPECT_GT(first.clangVersion.size, 0);

    for (auto capability : {"output:lto-ir", "output:ptx-text"})
    {
        bool found = false;

        for (size_t i = 0; i < first.capabilityCount; ++i)
        {
            found |= to_string(first.capabilities[i]) == capability;
        }

        EXPECT_TRUE(found) << capability;
    }
}

/**
 * @brief Keep public status values and their textual descriptions consistent.
 *
 * Pins existing ABI values and checks descriptions for artifact, toolchain, expression, and link failures.
 */
TEST(ApiMetadata, StatusDescriptions)
{
    // Existing numeric values form part of the public C ABI.
    static_assert(CLANGRTC_SUCCESS == 0 && CLANGRTC_INTERNAL_ERROR == 7);
    std::pair<clangrtcStatus, std::string_view> cases[]{{CLANGRTC_INVALID_ARTIFACT, "invalid artifact"},
                                                        {CLANGRTC_TOOLCHAIN_ERROR, "toolchain error"},
                                                        {CLANGRTC_INVALID_EXPRESSION, "invalid expression"},
                                                        {CLANGRTC_LINK_FAILED, "link failed"}};

    for (auto const &[status, expected] : cases)
    {
        EXPECT_EQ(to_string(clangrtcStatusString(status)), expected);
    }
}
} // namespace test
