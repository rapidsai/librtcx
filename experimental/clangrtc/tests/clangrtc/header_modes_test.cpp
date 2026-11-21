/* SPDX-License-Identifier: Apache-2.0 */

/**
 * @file header_modes_test.cpp
 * @brief Exercise device header selection, standard-library forwarding, and prepared-state compatibility.
 */

#include "support.hpp"
#include "test_config.hpp"

#include <format>
#include <tuple>

namespace test
{
/** @brief Compile direct requests against an explicitly selected real CCCL installation. */
class HeaderModesTest : public CompilationTest
{
  protected:
    /** @brief Add the selected CCCL headers to the ordinary CUDA test configuration. */
    void SetUp() override
    {
        CompilationTest::SetUp();

        if (CcclPath.empty() || CcclPath.ends_with("-NOTFOUND"))
        {
            GTEST_SKIP() << "Select real CCCL headers with CLANGRTC_TEST_CCCL_INCLUDE_DIR";
        }

        options.push_back(std::format("-I{}", CcclPath));
    }
};

/** @brief Minimal headers retain actual device intrinsics without exposing runtime and texture declarations. */
TEST_F(HeaderModesTest, MinimalKeepsIntrinsics)
{
    options.insert(options.end(), {"--minimal", "--emit=nvvm-ir-text"});
    auto result = compile(request(R"CUDA(
#if !defined(__CUDACC_RTC_MINIMAL__) || defined(__CUDACC_RTC__)
#error Incorrect device compiler identity
#endif
#include <cuda/std/cstdint>

                                      extern "C" __global__ void kernel(int *out)
                                      {
                                          uint3 position = threadIdx;
                                          dim3 dimensions = blockDim;
                                          atomicMax(out, int(position.x + dimensions.x + __popc(7)));
                                          __syncthreads();
                                      }
                                  )CUDA",
                                  {}),
                          options);
    ASSERT_EQ(result.status, CLANGRTC_SUCCESS) << result.log();
    EXPECT_NE(result.output().find("@kernel"), std::string::npos);

    for (auto declaration : {"cudaError_t value;", "cudaTextureObject_t value;", "cudaStream_t value;"})
    {
        SCOPED_TRACE(declaration);
        auto invalid = compile(request(declaration, {}), options);
        EXPECT_EQ(invalid.status, CLANGRTC_COMPILATION_FAILED) << invalid.log();
    }
}

/** @brief Real libcudacxx types and algorithms back std without importing a host C++ library. */
TEST_F(HeaderModesTest, ForwardsDeviceStandardLibrary)
{
    if (!CcclFreestanding)
    {
        GTEST_SKIP() << "The selected CCCL predates _CCCL_ENABLE_FREESTANDING";
    }

    options.insert(options.end(), {"--stdlib=libcudacxx", "--emit=nvvm-ir-text"});

    for (bool minimal : {false, true})
    {
        if (minimal)
        {
            options.push_back("--minimal");
        }

        auto result =
            compile(request(R"CUDA(
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <climits>
#include <type_traits>
#include <utility>
#include <chrono>
#include <array>
#include <tuple>
#include <optional>
#include <span>
#include <bit>
#include <functional>
#include <string_view>
#include <algorithm>
#include <cmath>
#include <initializer_list>
#if defined(__GLIBCXX__) || defined(_LIBCPP_VERSION) || defined(__CUDACC_RTC__)
#error Device forwarding imported a host C++ library or changed the compiler identity
#endif
                                static_assert(std::is_same_v<std::chrono::nanoseconds, cuda::std::chrono::nanoseconds>);
                                static_assert(std::is_same_v<std::array<int, 2>, cuda::std::array<int, 2>>);
                                static_assert(std::is_same_v<std::span<int>, cuda::std::span<int>>);

                                extern "C" __global__ void kernel(int *out)
                                {
                                    std::array<int, 2> values{1, 2};
                                    auto [first, second] = values;
                                    std::span<int> view(values);
                                    std::optional<int> value{3};
                                    auto tuple = std::make_tuple(*value, view[0]);
                                    auto list = {first, second};
                                    *out = std::get<0>(tuple) + std::min(first, second) + int(list.size()) + int(std::sqrt(float(view[1])));
                                }
                            )CUDA",
                            {}),
                    options);
        EXPECT_EQ(result.status, CLANGRTC_SUCCESS) << result.log();
    }
}

/** @brief Device header options reject native-host compilation and unsupported standard libraries. */
TEST_F(HeaderModesTest, RejectsInvalidHeaderModes)
{
    for (auto flag : {"--minimal", "--stdlib=libcudacxx", "--stdlib=unknown"})
    {
        auto result = compile(request("int value;", {}), {"--host-only", flag});
        EXPECT_EQ(result.status, CLANGRTC_INVALID_OPTION) << result.log();
    }
}

/** @brief Reuse each combination of header modes across all prepared compiler formats. */
class PreparedHeaderModesTest : public HeaderModesTest,
                                public ::testing::WithParamInterface<std::tuple<std::string, bool, bool>>
{
};

/** @brief Prepared files retain their header mode and reject a consumer selecting a different one. */
TEST_P(PreparedHeaderModesTest, RetainsHeaderMode)
{
    auto const &[kind, minimal, libcudacxx] = GetParam();

    if (libcudacxx && !CcclFreestanding)
    {
        GTEST_SKIP() << "The selected CCCL predates _CCCL_ENABLE_FREESTANDING";
    }

    options.push_back(std::format("--artifact={}", kind));
    options.push_back("--emit=nvvm-ir-text");

    if (minimal)
    {
        options.push_back("--minimal");
    }

    if (libcudacxx)
    {
        options.push_back("--stdlib=libcudacxx");
    }

    auto module = kind != "pch";
    auto source = module ? std::string("module;\n#include <clangrtc/device.hpp>\n") : std::string{};
    source.append(libcudacxx ? "#include <chrono>\n" : "#include <cuda/std/chrono>\n");
    source.append(R"CUDA(
        namespace prepared
        {
        template <class T> __device__ int value(T count)
        {
            return int(cuda::std::chrono::milliseconds(count).count());
        }
        } // namespace prepared
    )CUDA");

    if (module)
    {
        source.append("export module header_environment;\nexport namespace prepared { using ::prepared::value; }\n");
    }

    auto retained = to_view("&prepared::value<int>");
    auto producer = request(source, {&retained, 1});
    producer.source.path = to_view("/__clangrtc__/header_environment.cppm");
    auto prepared = prepare(producer, options);
    ASSERT_EQ(prepared.status, CLANGRTC_SUCCESS) << prepared.log();
    auto loaded = load(prepared.output());
    ASSERT_EQ(loaded.status, CLANGRTC_SUCCESS);
    auto consumer = std::format("{}extern \"C\" __global__ void kernel(int *out) {{ *out = prepared::value(7LL); }}",
                                module ? "import header_environment;\n" : "");
    auto input = request(consumer, {});
    input.artifact = loaded.artifact.get();
    auto result = compile(input, {});
    EXPECT_EQ(result.status, CLANGRTC_SUCCESS) << result.log();
    auto changed = options;
    changed.push_back(libcudacxx ? "--stdlib=host" : "--stdlib=libcudacxx");
    auto incompatible = compile(input, changed);
    EXPECT_EQ(incompatible.status, CLANGRTC_INCOMPATIBLE_ARTIFACT) << incompatible.log();
    changed = options;

    if (minimal)
    {
        std::erase(changed, "--minimal");
    }
    else
    {
        changed.push_back("--minimal");
    }

    incompatible = compile(input, changed);
    EXPECT_EQ(incompatible.status, CLANGRTC_INCOMPATIBLE_ARTIFACT) << incompatible.log();
}

INSTANTIATE_TEST_SUITE_P(Formats, PreparedHeaderModesTest,
                         ::testing::Combine(::testing::Values("pch", "bmi", "reduced-bmi"), ::testing::Bool(),
                                            ::testing::Bool()));
} // namespace test
