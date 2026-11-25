/* SPDX-License-Identifier: Apache-2.0 */

/**
 * @file options_test.cpp
 * @brief Test backend initialization, forwarded compiler options, and component version queries.
 *
 * Checks argument boundaries, workflow-reserved flags, and frontend-artifact compatibility when backend options change.
 */

#include "support.hpp"

namespace test
{
/**
 * @brief Allow repeated explicit backend initialization before version queries.
 *
 * Initializes twice and checks that the libNVVM version remains available from the initialized backend.
 */
TEST(BackendInitialization, ExplicitInitIsIdempotent)
{
    EXPECT_EQ(clangrtcInit(nullptr, nullptr), CLANGRTC_SUCCESS);
    EXPECT_EQ(clangrtcInit(nullptr, nullptr), CLANGRTC_SUCCESS);
    clangrtcString version{};
    ASSERT_EQ(clangrtcGetVersion(to_view("nvvm"), &version), CLANGRTC_SUCCESS);
    EXPECT_GT(version.size, 0);
}

/**
 * @brief Forward Clang options without splitting arguments or reordering definitions.
 *
 * Uses an include directory containing spaces, repeated macro overrides, and hidden visibility to verify the generated
 * host IR.
 */
TEST(ForwardedOptions, ClangPreservesOrderAndArgumentBoundaries)
{
    auto input = request(R"cpp(
#include "number.hpp"
extern "C" int answer() { return VALUE + EXTRA; }
)cpp",
                         {});
    clangrtcFile header{to_view("/__clangrtc__/headers with spaces/number.hpp"), to_view("#define EXTRA 2\n")};
    input.headers = &header;
    input.headerCount = 1;
    auto result = compile(input, {"--host-only", "--emit=llvm-ir-text", "--clang-option=-I",
                                  "--clang-option=/__clangrtc__/headers with spaces", "--clang-option=-DVALUE=1",
                                  "--clang-option=-UVALUE", "--clang-option", "-DVALUE=40",
                                  "--clang-option=-fvisibility=hidden"});
    ASSERT_EQ(result.status, CLANGRTC_SUCCESS) << result.log();
    EXPECT_NE(result.output().find("define hidden"), std::string::npos);
    EXPECT_NE(result.output().find("ret i32 42"), std::string::npos);
}

/** @brief A caller include directory overrides an SDK header while still forwarding to the real toolkit. */
TEST_F(CompilationTest, CallerIncludeDirectoriesPrecedeToolkitHeaders)
{
    auto input = request(R"CUDA(
#include <cuda_runtime_api.h>
#ifndef SELECTED_CALLER_HEADER
#error The toolkit header took precedence over the caller include directory
#endif
                             extern "C" __global__ void kernel()
                             {
                             }
                         )CUDA",
                         {});
    clangrtcFile header{to_view("/__clangrtc__/caller/cuda_runtime_api.h"),
                        to_view("#define SELECTED_CALLER_HEADER 1\n#include_next <cuda_runtime_api.h>\n")};
    input.headers = &header;
    input.headerCount = 1;
    options.push_back("-I/__clangrtc__/caller");
    auto result = compile(input, options);
    ASSERT_EQ(result.status, CLANGRTC_SUCCESS) << result.log();
}

/**
 * @brief Reject forwarded arguments that override the workflow or omit required values.
 *
 * Covers driver controls, plugins, response files, source filenames, and incomplete option forms without publishing
 * output.
 */
TEST(ForwardedOptions, RejectsPipelineOverridesAndIncompleteArguments)
{
    for (auto option : {"-o", "-x", "-c", "-target", "-Xclang", "-load", "-fsyntax-only", "-fplugin=bad.so",
                        "input.cpp", "@response", "-I", ""})
    {
        auto result =
            compile(request("int answer() {return 42;}", {}), {"--host-only", std::string("--clang-option=") + option});
        EXPECT_EQ(result.status, CLANGRTC_INVALID_OPTION) << option << result.log();
        EXPECT_TRUE(result.output().empty());
    }

    auto result = compile(request("int answer() {return 42;}", {}), {"--host-only", "--clang-option"});
    EXPECT_EQ(result.status, CLANGRTC_INVALID_OPTION);
}

/**
 * @brief Forward repeated NVVM options and preserve backend rejection diagnostics.
 *
 * Checks a floating-point instruction modifier, then supplies an unknown option and requires a failed result with no
 * output.
 */
TEST_F(CompilationTest, MultipleNvvmOptionsReachBackend)
{
    options.insert(options.end(), {"--ptx-text", "--nvvm-option=-ftz=1", "--nvvm-option", "-prec-div=0"});
    auto result = compile(request(R"cuda(
extern "C" __global__ void divide(float *out, float a, float b) { *out = a/b; }
)cuda",
                                  {}),
                          options);
    ASSERT_EQ(result.status, CLANGRTC_SUCCESS) << result.log();
    EXPECT_NE(result.output().find(".ftz.f32"), std::string::npos);
    options.push_back("--nvvm-option=-not-a-real-nvvm-option");
    auto failure = compile(request("extern \"C\" __global__ void kernel() {}", {}), options);
    EXPECT_EQ(failure.status, CLANGRTC_COMPILATION_FAILED) << failure.log();
    EXPECT_NE(failure.log().find("not-a-real-nvvm-option"), std::string::npos);
    EXPECT_TRUE(failure.output().empty());
}

/**
 * @brief Reserve NVVM architecture, backend selection, and LTO output flags for clangrtc.
 *
 * Attempts to forward each workflow-owned flag and requires invalid-option status before backend compilation.
 */
TEST_F(CompilationTest, NvvmCannotOverrideTargetOrOutput)
{
    for (auto option : {"-arch=compute_50", "-gen-lto", "-nvvm-latest"})
    {
        auto flags = options;
        flags.push_back(std::string("--nvvm-option=") + option);
        auto result = compile(request("extern \"C\" __global__ void kernel() {}", {}), flags);
        EXPECT_EQ(result.status, CLANGRTC_INVALID_OPTION) << result.log();
    }
}

/**
 * @brief Reuse frontend artifacts when only NVVM options change.
 *
 * Changes backend tuning successfully, then changes a frontend definition and requires an incompatible-artifact
 * failure.
 */
TEST_F(CompilationTest, BackendOptionsDoNotInvalidateFrontendArtifact)
{
    auto preparation = request("template<class T> __device__ T identity(T x) {return x;}", {});
    preparation.source.path = to_view("/__clangrtc__/options-prefix.cuh");
    auto flags = options;
    flags.insert(flags.end(), {"--artifact=pch", "--clang-option=-DFRONTEND=1", "--nvvm-option=-ftz=0"});
    auto prepared = prepare(preparation, flags);
    ASSERT_EQ(prepared.status, CLANGRTC_SUCCESS) << prepared.log();
    auto loaded = load(prepared.output());
    ASSERT_EQ(loaded.status, CLANGRTC_SUCCESS);
    auto input = request("extern \"C\" __global__ void kernel(int* out) {*out=identity(42);}", {});
    input.artifact = loaded.artifact.get();
    flags.back() = "--nvvm-option=-ftz=1";
    flags.push_back("--lto-ir");
    auto result = compile(input, flags);
    ASSERT_EQ(result.status, CLANGRTC_SUCCESS) << result.log();
    flags.push_back("--clang-option=-DFRONTEND=2");
    auto mismatch = compile(input, flags);
    EXPECT_EQ(mismatch.status, CLANGRTC_INCOMPATIBLE_ARTIFACT) << mismatch.log();
}

/**
 * @brief Report stable component versions and reject unsupported version-query arguments.
 *
 * Compares Clang/LLVM versions with compiler metadata and checks unknown components, empty names, and NULL outputs.
 */
TEST(VersionQueries, ReportsComponentsAndValidatesArguments)
{
    clangrtcCompilerInfo info{};
    ASSERT_EQ(clangrtcGetInfo(&info), CLANGRTC_SUCCESS);

    for (auto name : {"clang", "llvm", "nvvm"})
    {
        clangrtcString version{}, repeated{};
        auto status = clangrtcGetVersion(to_view(name), &version);
        ASSERT_EQ(status, CLANGRTC_SUCCESS) << name;
        EXPECT_FALSE(to_string(version).empty());
        EXPECT_NE(to_string(version).find('.'), std::string::npos);
        EXPECT_EQ(clangrtcGetVersion(to_view(name), &repeated), CLANGRTC_SUCCESS);
        EXPECT_EQ(to_string(version), to_string(repeated));

        if (std::string_view(name) == "clang")
        {
            EXPECT_EQ(to_string(version), to_string(info.clangVersion));
        }

        if (std::string_view(name) == "llvm")
        {
            EXPECT_EQ(to_string(version), to_string(info.llvmVersion));
        }
    }

    clangrtcString version = to_view("sentinel");
    EXPECT_EQ(clangrtcGetVersion(to_view("nvjitlink"), &version), CLANGRTC_INVALID_ARGUMENT);
    EXPECT_EQ(version.size, 0);
    EXPECT_EQ(version.data, nullptr);
    EXPECT_EQ(clangrtcGetVersion({}, &version), CLANGRTC_INVALID_ARGUMENT);
    EXPECT_EQ(clangrtcGetVersion(to_view("clang"), nullptr), CLANGRTC_INVALID_ARGUMENT);
}
} // namespace test
