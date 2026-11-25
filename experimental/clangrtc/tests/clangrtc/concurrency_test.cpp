/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* SPDX-License-Identifier: Apache-2.0 */

/**
 * @file concurrency_test.cpp
 * @brief Test concurrent compilation with shared artifacts and independent backend programs.
 *
 * Coordinates overlapping calls to detect accidental sharing of frontend state, diagnostics, or libNVVM programs.
 */

#include "support.hpp"

#include <future>
#include <latch>

namespace test
{
using ConcurrencyTest = CompilationTest;

/**
 * @brief Reuse one artifact concurrently without sharing frontend or diagnostic state.
 *
 * Starts overlapping consumers of a prepared PCH and requires independent successful results and callback observations.
 */
TEST_F(ConcurrencyTest, SharedArtifactHasIndependentFrontendAndCallbacks)
{
    options.push_back("--artifact=pch");
    auto prepared = prepare(request("template<class T> __device__ T identity(T x) { return x; }", {}), options);
    ASSERT_EQ(prepared.status, CLANGRTC_SUCCESS) << prepared.log();
    auto loaded = load(prepared.output());
    ASSERT_EQ(loaded.status, CLANGRTC_SUCCESS);
    std::latch ready(2);
    auto worker = [&] {
        unsigned diagnostics = 0;
        auto input = request(R"CUDA(
#warning concurrent compilation

                                 extern "C" __global__ void concurrent(int *out)
                                 {
                                     *out = identity(7);
                                 }
                             )CUDA",
                             {});
        input.source.path = to_view("/__clangrtc__/parallel.cu");
        input.artifact = loaded.artifact.get();
        ready.count_down();
        ready.wait();
        auto result = compile(
            input, {}, [](clangrtcDiagnostic const *, void *data) { ++*static_cast<unsigned *>(data); }, &diagnostics);
        return std::pair{std::move(result), diagnostics};
    };

    auto first = std::async(std::launch::async, worker);
    auto second = std::async(std::launch::async, worker);

    for (auto *future : {&first, &second})
    {
        auto [result, diagnostics] = future->get();
        EXPECT_EQ(result.status, CLANGRTC_SUCCESS) << result.log();
        EXPECT_FALSE(result.output().empty());
        EXPECT_GT(diagnostics, 0);
    }
}

/**
 * @brief Compile LTO and PTX concurrently using separate libNVVM programs.
 *
 * Runs both backend output modes in parallel and verifies that each call publishes its own nonempty result.
 */
TEST_F(ConcurrencyTest, IndependentBackendPrograms)
{
    auto worker = [&](char const *flag) {
        auto flags = options;
        flags.push_back(flag);
        return compile(request("extern \"C\" __global__ void kernel(int* out) {*out=42;}", {}), flags);
    };

    auto lto = std::async(std::launch::async, worker, "--lto-ir");
    auto ptx = std::async(std::launch::async, worker, "--ptx-text");

    for (auto *future : {&lto, &ptx})
    {
        auto result = future->get();
        EXPECT_EQ(result.status, CLANGRTC_SUCCESS) << result.log();
        EXPECT_FALSE(result.output().empty());
    }
}
} // namespace test
