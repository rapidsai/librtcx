/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* SPDX-License-Identifier: Apache-2.0 */

/**
 * @file toolchain_test.cpp
 * @brief Test diagnostics for missing compiler resources and malformed CUDA toolkit metadata.
 *
 * Uses deliberately invalid paths and version headers to verify toolchain-error status and useful diagnostics.
 */

#include "support.hpp"

#include <format>
#include <fstream>

namespace test
{
class MissingToolchainTest : public CompilationTest, public ::testing::WithParamInterface<char const *>
{
};

/**
 * @brief Name the missing CUDA toolkit or Clang resource path in toolchain diagnostics.
 *
 * Supplies a nonexistent directory through each supported path option and checks error status and empty output.
 */
TEST_P(MissingToolchainTest, IdentifiesMissingPath)
{
    options.push_back(std::format("{}={}/missing", GetParam(), temporary.path.string()));
    auto result = compile(request("extern \"C\" __global__ void kernel() {}", {}), options);
    EXPECT_EQ(result.status, CLANGRTC_TOOLCHAIN_ERROR);
    EXPECT_EQ(result.info().status, result.status);
    EXPECT_TRUE(result.output().empty());
    EXPECT_NE(result.log().find(temporary.path.string()), std::string::npos);
}

/** @brief Exercise missing CUDA toolkit and Clang resource directory overrides. */
INSTANTIATE_TEST_SUITE_P(Paths, MissingToolchainTest, ::testing::Values("--cuda-path", "--resource-dir"));

class InvalidToolkitTest : public CompilationTest, public ::testing::WithParamInterface<char const *>
{
};

/**
 * @brief Reject CUDA toolkit headers with missing or malformed version definitions.
 *
 * Creates a temporary runtime header from each invalid parameter and requires a toolchain error with diagnostics.
 */
TEST_P(InvalidToolkitTest, RejectsInvalidCudaVersionHeader)
{
    auto include = temporary.path / "include";
    std::filesystem::create_directories(include);
    {
        std::ofstream output(include / "cuda_runtime_api.h");
        output << GetParam();
    }

    options.push_back(std::format("--cuda-path={}", temporary.path.string()));
    auto result = compile(request("extern \"C\" __global__ void kernel() {}", {}), options);
    EXPECT_EQ(result.status, CLANGRTC_TOOLCHAIN_ERROR);
    EXPECT_TRUE(result.output().empty());
    EXPECT_FALSE(result.log().empty());
}

/** @brief Exercise missing and malformed CUDA runtime version definitions. */
INSTANTIATE_TEST_SUITE_P(Version, InvalidToolkitTest,
                         ::testing::Values("// missing version\n", "#define CUDART_VERSION invalid\n"));
} // namespace test
