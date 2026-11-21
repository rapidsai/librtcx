/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* SPDX-License-Identifier: Apache-2.0 */

/**
 * @file compilation_test.cpp
 * @brief Test device compilation, output ownership, source views, and kernel-symbol resolution.
 *
 * Exercises NVVM IR, NVIDIA LTO IR, and PTX generation without requiring a CUDA driver or GPU.
 */

#include "support.hpp"

#include <algorithm>
#include <cctype>

extern "C" int check_c_header(void);

namespace test
{
namespace
{
constexpr std::string_view Source = R"CUDA(
    __device__ __attribute__((used)) int retained = 0;

    extern "C" __global__ void kernel(int *out)
    {
        *out = retained + 7;
    }
)CUDA";

void expect_failure_location(std::string const &log)
{
    EXPECT_NE(log.find("clangrtc failure at "), std::string::npos);
    auto file = log.find("compiler.cpp:");
    ASSERT_NE(file, std::string::npos);
    auto line = file + std::string_view("compiler.cpp:").size();
    auto end = log.find(": ", line);
    ASSERT_NE(end, std::string::npos);
    ASSERT_GT(end, line);
    EXPECT_TRUE(std::all_of(log.begin() + line, log.begin() + end,
                            [](unsigned char character) { return std::isdigit(character); }));
}
} // namespace

/**
 * @brief Verify that the public header compiles and can be called from C.
 *
 * Runs the separately compiled C probe so C++ compilation cannot conceal incompatible C declarations.
 */
TEST(CHeader, CompilesAsC)
{
    EXPECT_EQ(check_c_header(), 0);
}

/**
 * @brief Preserve kernel definitions, used device globals, and NVVM metadata in textual IR.
 *
 * Compiles without optimization and checks both program symbols and the metadata required by the NVIDIA backend.
 */
TEST_F(CompilationTest, TextRetainsKernelUsedGlobalAndNvvmMetadata)
{
    options.insert(options.end(), {"--emit=nvvm-ir-text", "-O0"});
    auto result = compile(request(Source, {}), options);
    ASSERT_EQ(result.status, CLANGRTC_SUCCESS) << result.log();
    EXPECT_EQ(to_string(result.info().format), "nvvm-ir-text");
    auto ir = result.output();

    for (auto symbol : {"@kernel", "@retained", "@llvm.compiler.used", "nvvmir.version"})
    {
        EXPECT_NE(ir.find(symbol), std::string::npos) << symbol;
    }
}

/**
 * @brief Return length-delimited LLVM bitcode without truncating embedded NUL bytes.
 *
 * Checks the bitcode magic and confirms that the owned output contains binary data beyond C-string boundaries.
 */
TEST_F(CompilationTest, BitcodePreservesEmbeddedNullBytes)
{
    options.push_back("--emit=nvvm-ir-bitcode");
    auto result = compile(request(Source, {}), options);
    ASSERT_EQ(result.status, CLANGRTC_SUCCESS) << result.log();
    EXPECT_EQ(to_string(result.info().format), "nvvm-ir-bitcode");
    auto bytes = result.output();
    ASSERT_GT(bytes.size(), 4);
    EXPECT_EQ(bytes.substr(0, 4), std::string("BC\xc0\xde", 4));
    EXPECT_NE(bytes.find('\0'), std::string::npos);
}

/**
 * @brief Produce owned LTO and PTX results through libNVVM.
 *
 * Checks each output representation, PTX terminator removal, and result lifetime across a later compilation.
 */
TEST_F(CompilationTest, BackendOutputs)
{
    for (auto flag : {"--lto-ir", "--ptx-text"})
    {
        auto flags = options;
        flags.push_back(flag);
        auto result = compile(request(Source, {}), flags);
        ASSERT_EQ(result.status, CLANGRTC_SUCCESS) << result.log();
        EXPECT_EQ(to_string(result.info().format), std::string(flag).substr(2));
        auto bytes = result.output();
        ASSERT_GT(bytes.size(), 4);

        if (std::string_view(flag) == "--ptx-text")
        {
            EXPECT_NE(bytes.find(".version"), std::string::npos);
            EXPECT_NE(bytes.find("kernel"), std::string::npos);
            EXPECT_NE(bytes.back(), '\0');
        }
        else
        {
            EXPECT_NE(bytes.find('\0'), std::string::npos);
        }

        // Result ownership is independent of subsequent compilations.
        auto other = compile(request(Source, {}), options);
        ASSERT_EQ(other.status, CLANGRTC_SUCCESS) << other.log();
        EXPECT_EQ(result.output(), bytes);
    }
}

/**
 * @brief Attribute libNVVM rejection to the backend and discard partial output.
 *
 * Uses a target rejected by the backend and verifies status, diagnostics, and the absence of usable result bytes.
 */
TEST_F(CompilationTest, BackendFailureReportsStageAndClearsOutput)
{
    options.insert(options.end(), {"--lto-ir", "--gpu-architecture=compute_50"});
    bool backend_error = false;
    auto result = compile(
        request(Source, {}), options,
        [](clangrtcDiagnostic const *d, void *state) {
            if (to_string(d->stage) == "nvvm" && d->level == CLANGRTC_DIAGNOSTIC_ERROR)
            {
                *static_cast<bool *>(state) = true;
            }
        },
        &backend_error);
    EXPECT_EQ(result.status, CLANGRTC_COMPILATION_FAILED) << result.log();
    EXPECT_TRUE(backend_error);
    EXPECT_NE(result.log().find("nvvmCompileProgram"), std::string::npos);
    EXPECT_NE(result.log().find("compiler.cpp:"), std::string::npos);
    EXPECT_TRUE(result.output().empty());
    EXPECT_TRUE(to_string(result.info().format).empty());
    EXPECT_NE(result.log().find("compute_50"), std::string::npos);
}

/**
 * @brief Reject multiple explicit output selectors during compilation and preparation.
 *
 * Covers conflicting modes, equivalent aliases, and repeated identical flags. Failed
 * operations report invalid-option status and publish neither output bytes nor a format.
 */
TEST_F(CompilationTest, RejectsMultipleOutputSelectors)
{
    constexpr std::string_view selectors[]{
        "--lto-ir", "--ptx-text", "--emit=lto-ir", "--emit=ptx-text", "--emit=nvvm-ir-bitcode", "--emit=nvvm-ir-text"};

    for (auto first : selectors)
    {
        for (auto second : selectors)
        {
            SCOPED_TRACE(first);
            SCOPED_TRACE(second);
            auto flags = options;
            flags.emplace_back(first);
            flags.emplace_back(second);

            for (bool preparing : {false, true})
            {
                auto input = request(Source, {});
                auto result = preparing ? prepare(input, flags) : compile(input, flags);
                EXPECT_EQ(result.status, CLANGRTC_INVALID_OPTION) << result.log();
                EXPECT_EQ(result.info().status, result.status);
                EXPECT_TRUE(result.output().empty());
                EXPECT_TRUE(to_string(result.info().format).empty());
                EXPECT_NE(result.log().find("Only one output selector"), std::string::npos);
            }
        }
    }
}

/**
 * @brief Reject unsupported options with a diagnostic containing the failure call site.
 *
 * Checks invalid-option status, empty output, and a source filename with a numeric line location.
 */
TEST_F(CompilationTest, RejectsUnknownOptionWithCallsite)
{
    options.push_back("--not-a-real-option");
    auto result = compile(request(Source, {}), options);
    EXPECT_EQ(result.status, CLANGRTC_INVALID_OPTION);
    EXPECT_EQ(result.info().status, result.status);
    EXPECT_TRUE(result.output().empty());
    expect_failure_location(result.log());
}

/**
 * @brief Reject an empty source view before producing compiler output.
 *
 * Checks invalid-argument status and the internal call-site information carried by the diagnostic.
 */
TEST_F(CompilationTest, RejectsEmptySourceWithCallsite)
{
    auto result = compile(request({}, {}), options);
    EXPECT_EQ(result.status, CLANGRTC_INVALID_ARGUMENT);
    EXPECT_TRUE(result.output().empty());
    expect_failure_location(result.log());
}

/**
 * @brief Honor the source view length even when invalid bytes follow it.
 *
 * Appends an error directive outside an unterminated view and requires compilation of only the requested prefix.
 */
TEST_F(CompilationTest, SourceUsesExplicitLength)
{
    // The invalid suffix is outside the view, and there is no terminator at its end.
    std::vector<char> bytes(Source.begin(), Source.end());
    constexpr std::string_view suffix = "\n#error bytes beyond the view were read\n";

    bytes.insert(bytes.end(), suffix.begin(), suffix.end());
    auto input = request(std::string_view(bytes.data(), Source.size()), {});
    auto result = compile(input, options);
    EXPECT_EQ(result.status, CLANGRTC_SUCCESS) << result.log();
}

/**
 * @brief Pass embedded source NUL bytes to Clang instead of rejecting the view.
 *
 * Places a NUL inside a comment and verifies that the frontend can still compile the complete source buffer.
 */
TEST_F(CompilationTest, EmbeddedNullInCommentIsLeftToFrontend)
{
    auto source = std::string(Source);
    source.append("\n// embedded ");
    source.push_back('\0');
    source.append(" byte\n");
    auto result = compile(request(source, {}), options);
    EXPECT_EQ(result.status, CLANGRTC_SUCCESS) << result.log();
}

/**
 * @brief Instantiate a selected kernel template and report its expression and CUDA name.
 *
 * Selects an integer specialization and checks that one resolved linker symbol is returned with source metadata.
 */
TEST_F(CompilationTest, ResolvesTemplateEntry)
{
    auto slot = to_view("&generic<int>");
    auto input = request(R"CUDA(
                             template <class T> __global__ void generic(T *out)
                             {
                                 *out = T{9};
                             }
                         )CUDA",
                         {&slot, 1});
    auto result = compile(input, options);
    ASSERT_EQ(result.status, CLANGRTC_SUCCESS) << result.log();
    auto info = result.info();
    ASSERT_EQ(info.symbolCount, 1);
    EXPECT_EQ(to_string(info.symbols[0].expression), "&generic<int>");
    EXPECT_NE(info.symbols[0].name.size, 0);
    EXPECT_EQ(to_string(info.symbols[0].cudaName), "generic<int>");
}
} // namespace test
