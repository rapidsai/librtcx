/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* SPDX-License-Identifier: Apache-2.0 */

/**
 * @file artifacts_test.cpp
 * @brief Test artifact dependency capture, expression retention, reuse, and package validation.
 *
 * Runs the same contracts across all supported artifact formats after the original headers have been removed.
 */

#include "artifact_fixture.hpp"

#include <format>

namespace test
{
/**
 * @brief Keep packaged dependencies and preparation-time expression metadata in each artifact.
 *
 * Inspects the loaded artifact after fixture setup has removed the original files and compares retained symbols and
 * kernel names.
 */
TEST_P(ArtifactTest, OwnsDependencyClosureAndPreparationSlots)
{
    clangrtcArtifactInfo info{};
    ASSERT_EQ(clangrtcGetArtifactInfo(artifact.get(), &info), CLANGRTC_SUCCESS);
    EXPECT_GT(info.dependencyCount, 2);
    EXPECT_EQ(info.packageSize, package.size());
    EXPECT_EQ(to_string(info.format), GetParam());
    ASSERT_EQ(info.symbolCount, preparedExpressions.size());

    for (size_t slot = 0; slot < info.symbolCount; ++slot)
    {
        EXPECT_EQ(to_string(info.symbols[slot].expression), preparedExpressions[slot]);
        EXPECT_EQ(to_string(info.symbols[slot].name), preparedNames[slot]);
        EXPECT_EQ(to_string(info.symbols[slot].cudaName), slot == 3 ? "example::prepared_kernel<int>" : "");
    }
}

/**
 * @brief Instantiate new consumers without changing the artifact preparation slots.
 *
 * Compiles integer and double consumers with and without request expressions and verifies that prepared definitions
 * remain retained.
 */
TEST_P(ArtifactTest, InstantiatesNewTypesWithIndependentCompilationSlots)
{
    for (auto type : {"int", "double"})
    {
        SCOPED_TRACE(type);
        auto source = import();
        // clang-format off
        source.append(std::format(R"CUDA(
            extern "C" __global__ void kernel({0}* out)
            {{
                *out = example::scale({0}{{5}});
            }}
        )CUDA", type));
        // clang-format on
        clangrtcString slots[]{to_view("&kernel"), to_view("&example::scale<double>")};
        auto input = request(isSource() ? std::string_view{} : std::string_view(source), slots);
        input.artifact = artifact.get();
        auto result = compile(input, {});
        ASSERT_EQ(result.status, CLANGRTC_SUCCESS) << result.log();
        auto info = result.info();
        ASSERT_EQ(info.symbolCount, 2);
        EXPECT_EQ(to_string(info.symbols[0].expression), "&kernel");
        EXPECT_EQ(to_string(info.symbols[0].name), "kernel");
        EXPECT_EQ(to_string(info.symbols[0].cudaName), "kernel");
        EXPECT_EQ(to_string(info.symbols[1].expression), "&example::scale<double>");
        EXPECT_EQ(info.symbols[1].cudaName.size, 0);
        auto ir = result.output();
        EXPECT_GT(ir.size(), 100);

        for (auto const &name : preparedNames)
        {
            EXPECT_NE(ir.find(name), std::string::npos) << name;
        }

        input.expressions = nullptr;
        input.expressionCount = 0;
        auto reused = compile(input, {});
        ASSERT_EQ(reused.status, CLANGRTC_SUCCESS) << reused.log();
        EXPECT_EQ(reused.info().symbolCount, 0);
    }
}

/**
 * @brief Retain preparation-time symbols when consumers append source without selecting expressions.
 *
 * Checks the compiled IR and then reinspects the immutable artifact metadata for its original expression-to-symbol
 * mappings.
 */
TEST_P(ArtifactTest, AdditionalSourcePreservesArtifactSlots)
{
    auto source = import();
    source.append(R"CUDA(
        extern "C" __global__ void consumer(int *out)
        {
            *out = example::scale(11);
        }
    )CUDA");
    auto slot = to_view("&consumer");
    auto input = request(source, {&slot, 1});
    input.artifact = artifact.get();
    auto result = compile(input, {});
    ASSERT_EQ(result.status, CLANGRTC_SUCCESS) << result.log();
    auto info = result.info();
    ASSERT_EQ(info.symbolCount, 1);
    EXPECT_EQ(to_string(info.symbols[0].name), "consumer");
    clangrtcArtifactInfo metadata{};
    ASSERT_EQ(clangrtcGetArtifactInfo(artifact.get(), &metadata), CLANGRTC_SUCCESS);
    ASSERT_EQ(metadata.symbolCount, preparedNames.size());

    for (size_t slot = 0; slot < metadata.symbolCount; ++slot)
    {
        EXPECT_EQ(to_string(metadata.symbols[slot].name), preparedNames[slot]);
    }
}

/**
 * @brief Generate LTO and PTX from consumers of prepared artifacts.
 *
 * Selects a consumer kernel and checks its resolved symbol and CUDA source name in both backend output modes.
 */
TEST_P(ArtifactTest, EmitsBackendOutputs)
{
    for (auto flag : {"--lto-ir", "--ptx-text"})
    {
        auto source = import();
        source.append(R"CUDA(
            extern "C" __global__ void backend_kernel(int *out)
            {
                *out = example::scale(5);
            }
        )CUDA");
        auto slot = to_view("&backend_kernel");
        auto input = request(source, {&slot, 1});
        input.artifact = artifact.get();
        auto flags = options;
        std::erase(flags, "--emit=nvvm-ir-text");
        flags.push_back(flag);
        auto result = compile(input, flags);
        ASSERT_EQ(result.status, CLANGRTC_SUCCESS) << result.log();
        EXPECT_EQ(to_string(result.info().format), std::string(flag).substr(2));
        ASSERT_EQ(result.info().symbolCount, 1);
        EXPECT_EQ(to_string(result.info().symbols[0].name), "backend_kernel");
        EXPECT_EQ(to_string(result.info().symbols[0].cudaName), "backend_kernel");
        EXPECT_FALSE(result.output().empty());
    }
}

/**
 * @brief Reject consumer options incompatible with an artifact semantic configuration.
 *
 * Changes the C++ language standard and requires incompatible-artifact status with diagnostics and no output.
 */
TEST_P(ArtifactTest, RejectsSemanticOptionMismatch)
{
    options.push_back("-std=c++17");
    auto input = request("extern \"C\" __global__ void other() {}", {});
    input.artifact = artifact.get();
    auto result = compile(input, options);
    EXPECT_EQ(result.status, CLANGRTC_INCOMPATIBLE_ARTIFACT);
    EXPECT_TRUE(result.output().empty());
    EXPECT_FALSE(result.log().empty());
}

/**
 * @brief Reject malformed serialized packages without returning artifact handles.
 *
 * Exercises truncated data, trailing bytes, and a changed payload byte against each artifact format.
 */
TEST_P(ArtifactTest, RejectsTruncatedTrailingAndCorruptPackages)
{
    auto truncated = load(package.substr(0, package.size() / 2));
    EXPECT_EQ(truncated.status, CLANGRTC_INVALID_ARTIFACT);
    EXPECT_EQ(truncated.artifact.get(), nullptr);
    auto trailing = package;
    trailing.push_back('x');
    auto extra = load(trailing);
    EXPECT_EQ(extra.status, CLANGRTC_INVALID_ARTIFACT);
    EXPECT_EQ(extra.artifact.get(), nullptr);
    auto corrupt = package;
    corrupt.back() ^= 1;
    auto changed = load(corrupt);
    EXPECT_EQ(changed.status, CLANGRTC_INVALID_ARTIFACT);
    EXPECT_EQ(changed.artifact.get(), nullptr);
}

/**
 * @brief Validate expression selections during artifact preparation.
 *
 * Uses virtual header data after deleting the original files and rejects nondeclarations, host symbols, missing names,
 * and undefined device declarations.
 */
TEST_P(ArtifactTest, RejectsInvalidPreparationExpressions)
{
    auto source = prefix;
    // Use a virtual dependency because SetUp removed the original directory.
    auto headerPath = (temporary.path / "scale.hpp").string();
    clangrtcFile header{to_view(headerPath), to_view("#define SCALE 3\n")};
    source.append(R"CUDA(
        __host__ int host_function()
        {
            return 0;
        }

        __device__ int undefined_function();
        extern __device__ int undefined_variable;
    )CUDA");
    auto flags = options;
    flags.push_back(std::format("--artifact={}", GetParam()));

    for (auto expression : {"42", "&missing", "&host_function", "&undefined_function", "&undefined_variable"})
    {
        SCOPED_TRACE(expression);
        auto slot = to_view(expression);
        auto input = request(source, {&slot, 1});
        input.source.path = to_view(prefixPath);
        input.headers = &header;
        input.headerCount = 1;
        auto result = prepare(input, flags);
        EXPECT_EQ(result.status, CLANGRTC_INVALID_EXPRESSION) << result.log();
        EXPECT_TRUE(result.output().empty());
        EXPECT_FALSE(result.log().empty());
    }
}

/** @brief Run the artifact contracts for PCH, BMI, reduced-BMI, source, and preprocessed-source packages. */
INSTANTIATE_TEST_SUITE_P(Formats, ArtifactTest,
                         ::testing::Values("pch", "bmi", "reduced-bmi", "cuda", "preprocessed-cuda"),
                         [](auto const &info) {
                             auto name = info.param;

                             for (auto &character : name)
                             {
                                 if (character == '-')
                                 {
                                     character = '_';
                                 }
                             }

                             return name;
                         });

/**
 * @brief Reject input that does not contain a serialized artifact package.
 *
 * Checks invalid-artifact status and a NULL artifact handle for a short arbitrary byte sequence.
 */
TEST(ArtifactLoading, RejectsNonPackageBytes)
{
    auto result = load("broken");
    EXPECT_EQ(result.status, CLANGRTC_INVALID_ARTIFACT);
    EXPECT_EQ(result.artifact.get(), nullptr);
}
} // namespace test
