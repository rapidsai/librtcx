/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* SPDX-License-Identifier: Apache-2.0 */

/**
 * @file profiles_test.cpp
 * @brief Test runtime-compilation profile macros and explicit freestanding configuration.
 *
 * Verifies which preprocessor definitions are supplied by the profile and which remain under caller control.
 */

#include "support.hpp"

namespace test
{
using ProfileTest = CompilationTest;

/**
 * @brief Preserve Clang compiler identity without implicitly enabling CCCL freestanding mode.
 *
 * Compiles source containing preprocessor assertions for both the required and intentionally absent definitions.
 */
TEST_F(ProfileTest, RuntimeProfilePreservesCompilerIdentity)
{
    options.push_back("--profile=nvrtc");
    options.push_back("--device-int128");
    auto result = compile(request(R"CUDA(
#ifdef __CUDACC_RTC__
#error "Clang must not identify as NVRTC"
#endif
#ifdef __CUDACC_RTC_INT128__
#error "Integer support must not imply NVRTC identity"
#endif
#ifdef _CCCL_ENABLE_FREESTANDING
#error "Freestanding mode must be requested explicitly"
#endif
                                      static_assert(sizeof(__int128) == 16);

                                      extern "C" __global__ void kernel(int *out)
                                      {
                                          *out = 1;
                                      }
                                  )CUDA",
                                  {}),
                          options);
    EXPECT_EQ(result.status, CLANGRTC_SUCCESS) << result.log();
}

/**
 * @brief Preserve an explicitly requested CCCL freestanding definition.
 *
 * Uses a source preprocessor check to verify that the caller-provided macro reaches Clang unchanged.
 */
TEST_F(ProfileTest, ForwardsExplicitFreestandingMacro)
{
    options.push_back("-D_CCCL_ENABLE_FREESTANDING=1");
    auto result = compile(request(R"CUDA(
#if !defined(_CCCL_ENABLE_FREESTANDING) || _CCCL_ENABLE_FREESTANDING != 1
#error "Explicit freestanding definition was not forwarded"
#endif
                                      extern "C" __global__ void kernel(int *out)
                                      {
                                          *out = 2;
                                      }
                                  )CUDA",
                                  {}),
                          options);
    EXPECT_EQ(result.status, CLANGRTC_SUCCESS) << result.log();
}
} // namespace test
