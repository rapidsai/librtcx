/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file host_test.cpp
 * @brief Test host C/C++ compilation and reuse of host PCH and module artifacts.
 *
 * Checks native LLVM IR and relocatable objects without depending on a CUDA toolkit installation.
 */

#include "support.hpp"

#include <fstream>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Object/ObjectFile.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/TargetParser/Host.h>

namespace test
{
/**
 * @brief Compile C to native LLVM bitcode without a working CUDA installation.
 *
 * Parses the result to check the host triple, expected function, and absence of NVVM metadata.
 */
TEST(HostCompilation, DefaultsToNativeBitcodeWithoutCuda)
{
    auto result = compile(request("int answer(void) { return 42; }", {}), {"--source=c", "--cuda-path=/missing/cuda"});
    ASSERT_EQ(result.status, CLANGRTC_SUCCESS) << result.log();
    EXPECT_EQ(to_string(result.info().format), "llvm-ir-bitcode");
    auto bytes = result.output();
    llvm::LLVMContext context;
    auto module = llvm::parseBitcodeFile(llvm::MemoryBufferRef(bytes, "host"), context);
    ASSERT_TRUE(static_cast<bool>(module)) << llvm::toString(module.takeError());
    EXPECT_EQ((*module)->getTargetTriple().str(), llvm::sys::getDefaultTargetTriple());
    EXPECT_EQ((*module)->getNamedMetadata("nvvmir.version"), nullptr);
    EXPECT_NE((*module)->getFunction("answer"), nullptr);
}

/**
 * @brief Support standard C++ headers and template selections during host compilation.
 *
 * Instantiates a function using a standard container and checks its resolved symbol in host IR without NVVM metadata.
 */
TEST(HostCompilation, CppHeadersTemplatesAndExpressionSlots)
{
    auto entry = to_view("&answer<int>");
    auto result = compile(request(R"cpp(
#include <vector>
template<class T> T answer(T x) { std::vector<T> v{x, T{2}}; return v[0]+v[1]; }
)cpp",
                                  {&entry, 1}),
                          {"--host-only", "--emit=llvm-ir-text"});
    ASSERT_EQ(result.status, CLANGRTC_SUCCESS) << result.log();
    auto info = result.info();
    EXPECT_EQ(to_string(info.format), "llvm-ir-text");
    ASSERT_EQ(info.symbolCount, 1);
    EXPECT_NE(result.output().find(to_string(info.symbols[0].name)), std::string::npos);
    EXPECT_EQ(result.output().find("nvvmir.version"), std::string::npos);
}

/**
 * @brief Emit a native object suitable for ordinary host linking.
 *
 * Parses the returned bytes with LLVM object readers and verifies that the result is relocatable.
 */
TEST(HostCompilation, EmitsNativeRelocatableObject)
{
    auto result = compile(request("extern \"C\" int answer() {return 42;}", {}), {"--source=c++", "--emit=object"});
    ASSERT_EQ(result.status, CLANGRTC_SUCCESS) << result.log();
    EXPECT_EQ(to_string(result.info().format), "object");
    auto bytes = result.output();
    auto object = llvm::object::ObjectFile::createObjectFile(llvm::MemoryBufferRef(bytes, "host"));
    ASSERT_TRUE(static_cast<bool>(object)) << llvm::toString(object.takeError());
    EXPECT_TRUE((*object)->isRelocatableObject());
}

/**
 * @brief Reject CUDA-only options and unsupported artifact combinations in host mode.
 *
 * Covers output and language conflicts, C module preparation, and malformed C source with no successful output.
 */
TEST(HostCompilation, RejectsInvalidCombinations)
{
    for (auto output : {"ptx-text", "lto-ir", "nvvm-ir-text"})
    {
        auto result = compile(request("int x;", {}), {"--host-only", std::string("--emit=") + output});
        EXPECT_EQ(result.status, CLANGRTC_INVALID_OPTION);
        EXPECT_TRUE(result.output().empty());
    }

    for (auto conflict : {"--source=cuda", "--gpu-architecture=sm_86", "--source=preprocessed-cuda"})
    {
        auto result = compile(request("int x;", {}), {"--host-only", conflict});
        EXPECT_EQ(result.status, CLANGRTC_INVALID_OPTION);
        EXPECT_TRUE(result.output().empty());
    }

    auto artifact = prepare(request("int x;", {}), {"--source=c", "--artifact=bmi"});
    EXPECT_EQ(artifact.status, CLANGRTC_UNSUPPORTED);
    auto result = compile(request("this is invalid C", {}), {"--source=c"});
    EXPECT_EQ(result.status, CLANGRTC_COMPILATION_FAILED);
    EXPECT_TRUE(result.output().empty());
}

/**
 * @brief Reject repeated or conflicting host output selectors.
 *
 * Covers every pair of supported host formats and verifies that invalid requests
 * return no output or selected format instead of accepting the final selector.
 */
TEST(HostCompilation, RejectsMultipleOutputSelectors)
{
    for (auto first : {"--emit=llvm-ir-bitcode", "--emit=llvm-ir-text", "--emit=object"})
    {
        for (auto second : {"--emit=llvm-ir-bitcode", "--emit=llvm-ir-text", "--emit=object"})
        {
            SCOPED_TRACE(first);
            SCOPED_TRACE(second);
            auto result = compile(request("int answer() { return 42; }", {}), {"--host-only", first, second});
            EXPECT_EQ(result.status, CLANGRTC_INVALID_OPTION) << result.log();
            EXPECT_TRUE(result.output().empty());
            EXPECT_TRUE(to_string(result.info().format).empty());
            EXPECT_NE(result.log().find("Only one output selector"), std::string::npos);
        }
    }
}

/**
 * @brief Preserve C ABI names when retaining C functions and variables.
 *
 * Checks resolved names and emitted definitions, then verifies that a missing C identifier is an invalid expression.
 */
TEST(HostCompilation, CNamesAreUnmangledAndRetained)
{
    clangrtcString entries[]{to_view("answer"), to_view("value")};
    auto result = compile(request("static int value = 42; static int answer(void) { return value; }", entries),
                          {"--source=c", "--emit=llvm-ir-text", "-O3"});
    ASSERT_EQ(result.status, CLANGRTC_SUCCESS) << result.log();
    auto info = result.info();
    ASSERT_EQ(info.symbolCount, 2);
    EXPECT_EQ(to_string(info.symbols[0].name), "answer");
    EXPECT_EQ(to_string(info.symbols[1].name), "value");
    EXPECT_NE(result.output().find("@answer"), std::string::npos);
    EXPECT_NE(result.output().find("@value"), std::string::npos);
    auto invalid = to_view("missing");
    auto failure = compile(request("int answer(void) {return 42;}", {&invalid, 1}), {"--source=c"});
    EXPECT_EQ(failure.status, CLANGRTC_INVALID_EXPRESSION) << failure.log();
    EXPECT_TRUE(failure.output().empty());
}

class HostArtifactTest : public ::testing::TestWithParam<char const *>
{
};

/**
 * @brief Reuse host artifacts after their original header files have been removed.
 *
 * Compiles new integer and double consumers to bitcode, textual IR, and objects while retaining prepared entries and
 * rejecting a language mismatch.
 */
TEST_P(HostArtifactTest, ReusesPackagedHeadersAndPreparedEntries)
{
    TemporaryDirectory temporary;
    auto header = temporary.path / "host.hpp";
    {
        std::ofstream file(header);
        file << "#define FACTOR 3\n";
    }

    bool module = std::string_view(GetParam()) != "pch";
    std::string prefix = module ? "module;\n" : "";
    prefix += "#include \"" + header.string() + "\"\n#include <array>\n";

    if (module)
    {
        prefix += "export module host_sample;\nexport ";
    }

    prefix += R"cpp(
        namespace sample {
            template<class T> T scale(T x) {std::array<T, 1> values{x}; return values[0]*FACTOR;}
        }
    )cpp";
    auto preparedEntry = to_view("&sample::scale<int>");
    Options flags{"--host-only", std::string("--artifact=") + GetParam(), "-O3"};
    auto preparation = request(prefix, {&preparedEntry, 1});
    preparation.source.path = to_view("/__clangrtc__/host-prefix.hpp");
    auto prepared = prepare(preparation, flags);
    ASSERT_EQ(prepared.status, CLANGRTC_SUCCESS) << prepared.log();
    auto loaded = load(prepared.output());
    ASSERT_EQ(loaded.status, CLANGRTC_SUCCESS);
    clangrtcArtifactInfo artifactInfo{};
    ASSERT_EQ(clangrtcGetArtifactInfo(loaded.artifact.get(), &artifactInfo), CLANGRTC_SUCCESS);
    EXPECT_EQ(to_string(artifactInfo.target), llvm::sys::getDefaultTargetTriple());
    ASSERT_EQ(artifactInfo.symbolCount, 1);
    auto preparedName = to_string(artifactInfo.symbols[0].name);
    std::filesystem::remove_all(temporary.path);

    for (auto type : {"int", "double"})
    {
        std::string source = module ? "import host_sample;\n" : "";
        source += "extern \"C\" " + std::string(type) + " answer(" + type + " x) {return sample::scale(x);}";
        auto entry = to_view("&answer");
        auto input = request(source, {&entry, 1});
        input.artifact = loaded.artifact.get();
        auto result = compile(input, {});
        ASSERT_EQ(result.status, CLANGRTC_SUCCESS) << result.log();
        EXPECT_EQ(to_string(result.info().format), "llvm-ir-bitcode");
        ASSERT_EQ(result.info().symbolCount, 1);
        EXPECT_EQ(to_string(result.info().symbols[0].name), "answer");
        auto outputFlags = flags;
        outputFlags.push_back("--emit=llvm-ir-text");
        auto textResult = compile(input, outputFlags);
        ASSERT_EQ(textResult.status, CLANGRTC_SUCCESS) << textResult.log();
        EXPECT_NE(textResult.output().find(preparedName), std::string::npos);
        EXPECT_EQ(textResult.output().find("nvvmir.version"), std::string::npos);
        outputFlags.back() = "--emit=object";
        auto objectResult = compile(input, outputFlags);
        ASSERT_EQ(objectResult.status, CLANGRTC_SUCCESS) << objectResult.log();
        EXPECT_EQ(to_string(objectResult.info().format), "object");
    }

    auto bad = request("int x;", {});
    bad.artifact = loaded.artifact.get();
    auto mismatch = compile(bad, {"--source=c"});
    EXPECT_EQ(mismatch.status, CLANGRTC_INCOMPATIBLE_ARTIFACT) << mismatch.log();
}

/** @brief Run host artifact reuse for PCH, BMI, and reduced-BMI formats. */
INSTANTIATE_TEST_SUITE_P(Formats, HostArtifactTest, ::testing::Values("pch", "bmi", "reduced-bmi"));

/**
 * @brief Reuse a C precompiled header while preserving unmangled consumer symbols.
 *
 * Prepares C functions and data, loads the artifact, and checks the symbol of a new consumer compiled from it.
 */
TEST(HostCompilation, CPrecompiledHeaderPreservesNames)
{
    clangrtcString entries[]{to_view("answer"), to_view("value")};
    auto preparation = request("static int value=40; int answer(int x) {return value+x;}", entries);
    preparation.source.path = to_view("/__clangrtc__/host-prefix.h");
    auto prepared = prepare(preparation, {"--source=c", "--artifact=pch"});
    ASSERT_EQ(prepared.status, CLANGRTC_SUCCESS) << prepared.log();
    auto loaded = load(prepared.output());
    ASSERT_EQ(loaded.status, CLANGRTC_SUCCESS);
    auto name = to_view("consumer");
    auto input = request("int consumer(void) {return answer(2);}", {&name, 1});
    input.artifact = loaded.artifact.get();
    auto result = compile(input, {});
    ASSERT_EQ(result.status, CLANGRTC_SUCCESS) << result.log();
    ASSERT_EQ(result.info().symbolCount, 1);
    EXPECT_EQ(to_string(result.info().symbols[0].name), "consumer");
}

} // namespace test
