/* SPDX-License-Identifier: Apache-2.0 */

/**
 * @file curtc_gpu_test.cpp
 * @brief Launch multiple cached kernels with runtime-linked device functions on a CUDA GPU.
 *
 * Uses the real driver and CUDA runtime to check execution results, repeated-request reuse, and library lifetime after
 * cache destruction.
 */

#include "gpu_support.hpp"
#include "support.hpp"
#include "test_config.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <curtc.h>
#include <format>
#include <gtest/gtest.h>

namespace
{
clangrtcString to_view(std::string_view text)
{
    return {text.data(), text.size()};
}

/** @brief Finalize and execute real libcudacxx-backed kernels with both full and minimal device headers. */
TEST(CurtcGpu, DeviceHeaderModesExecute)
{
    if (!test::CcclFreestanding)
    {
        GTEST_SKIP() << "Select CCCL with _CCCL_ENABLE_FREESTANDING support";
    }

    ASSERT_EQ(cudaSetDevice(test::GpuOrdinal), cudaSuccess);
    ASSERT_EQ(cudaFree(nullptr), cudaSuccess);
    curtcConfiguration configuration{};
    ASSERT_EQ(curtcDefaultConfiguration(&configuration), CLANGRTC_SUCCESS);
    configuration.diskCache = false;
    curtcCache rawCache{};
    ASSERT_EQ(curtcCreateCache(&configuration, &rawCache), CLANGRTC_SUCCESS);
    std::unique_ptr<curtcCacheImpl, decltype(&curtcDestroyCache)> cache(rawCache, curtcDestroyCache);
    auto source = std::string_view(R"CUDA(
#include <array>
#include <chrono>
#include <cmath>

        extern "C" __global__ void kernel(int *out)
        {
            std::array<int, 2> values{20, 22};
            auto count = std::chrono::milliseconds(values[0] + values[1]).count();

            if (std::sqrt(double(values[0] + 5)) != 5.0 || !std::isfinite(double(count)) ||
                !std::isnan(__builtin_bit_cast(float, unsigned(threadIdx.x) | 0x7fc00000u)) ||
                !std::isinf(__builtin_inff()) || !::isfinite(7) || ::isnan(7) || ::isinf(7))
            {
                return;
            }

            atomicMax(out, int(count));
        }
    )CUDA");
    auto target = test::gpu_target();
    test::Options options{std::format("--cuda-path={}", test::CudaPath), "--stdlib=libcudacxx",
                          std::format("-I{}", test::CcclPath)};
    curtcKernelSelection selection{to_view("&kernel"), {}};
    curtcInput input{};
    input.type = CURTC_INPUT_TYPE_CUDA;
    input.source.source = {to_view("/__gpu__/forwarding.cu"), to_view(source)};
    input.entries = &selection;
    input.entryCount = 1;
    curtcRequest request{};
    request.inputs = &input;
    request.inputCount = 1;
    request.target = to_view(target);

    for (bool minimal : {false, true})
    {
        if (minimal)
        {
            options.push_back("--minimal");
        }

        auto flags = test::to_views(options);
        input.options = flags.data();
        input.optionCount = flags.size();
        std::string log;
        auto diagnostic = [](clangrtcDiagnostic const *message, void *context) {
            static_cast<std::string *>(context)->append(message->message.data, message->message.size);
        };
        curtcLibrary raw{};
        ASSERT_EQ(curtcGet(cache.get(), &request, diagnostic, &log, &raw, nullptr), CLANGRTC_SUCCESS) << log;
        std::unique_ptr<curtcLibraryImpl, decltype(&curtcReleaseLibrary)> library(raw, curtcReleaseLibrary);
        CUkernel kernel{};
        ASSERT_EQ(curtcGetKernel(library.get(), to_view("kernel"), &kernel), CLANGRTC_SUCCESS);
        test::launch(kernel, 42);
    }
}

/** @brief Changed request-local CUDA functions and kernels execute correctly against the same prepared environment. */
TEST(CurtcGpu, PreparedArtifactsIsolateSameNameRequestDefinitions)
{
    ASSERT_EQ(cudaSetDevice(test::GpuOrdinal), cudaSuccess);
    ASSERT_EQ(cudaFree(nullptr), cudaSuccess);
    auto target = test::gpu_target();

    for (auto kind : {"pch", "bmi", "reduced-bmi"})
    {
        SCOPED_TRACE(kind);
        auto prefix = std::string("template<class T> __device__ T twice(T x) { return x + x; }\n");
        auto producer = std::string_view(kind) == "pch"
                            ? prefix
                            : std::format("module;\n#include <clangrtc/device.hpp>\n"
                                          "export module gpu_environment;\nexport {{ {} }}\n",
                                          prefix);
        auto preparation = test::request(producer, {});
        preparation.source.path = to_view("/__gpu__/environment.cppm");
        test::Options options{std::format("--cuda-path={}", test::CudaPath),
                              std::format("--gpu-architecture={}", target), std::format("--artifact={}", kind)};
        auto prepared = test::prepare(preparation, options);
        ASSERT_EQ(prepared.status, CLANGRTC_SUCCESS) << prepared.log();
        auto loaded = test::load(prepared.output());
        ASSERT_EQ(loaded.status, CLANGRTC_SUCCESS);
        curtcConfiguration configuration{};
        ASSERT_EQ(curtcDefaultConfiguration(&configuration), CLANGRTC_SUCCESS);
        configuration.diskCache = false;
        curtcCache rawCache{};
        ASSERT_EQ(curtcCreateCache(&configuration, &rawCache), CLANGRTC_SUCCESS);
        std::unique_ptr<curtcCacheImpl, decltype(&curtcDestroyCache)> cache(rawCache, curtcDestroyCache);
        curtcKernelSelection selection{to_view("&kernel"), {}};
        curtcInput input{};
        input.type = CURTC_INPUT_TYPE_ARTIFACT;
        input.source.artifact = loaded.artifact.get();
        input.entries = &selection;
        input.entryCount = 1;
        curtcRequest request{};
        request.inputs = &input;
        request.inputCount = 1;
        request.target = to_view(target);

        for (int value : {3, 7, 11})
        {
            auto imports = std::string_view(kind) == "pch" ? "" : "import gpu_environment;\n";
            auto source = std::format("{}extern \"C\" __device__ int udf(int x) {{ return x + {}; }}\n"
                                      "extern \"C\" __global__ void kernel(int *out) "
                                      "{{ *out = twice(udf(int(threadIdx.x))); }}\n",
                                      imports, value);
            input.source.source = {to_view("/__gpu__/consumer.cu"), to_view(source)};
            curtcLibrary raw{};
            std::string log;
            auto diagnostic = [](clangrtcDiagnostic const *message, void *context) {
                static_cast<std::string *>(context)->append(message->message.data, message->message.size);
            };
            curtcCompileStatistics statistics{};
            ASSERT_EQ(curtcGet(cache.get(), &request, diagnostic, &log, &raw, &statistics), CLANGRTC_SUCCESS) << log;
            std::unique_ptr<curtcLibraryImpl, decltype(&curtcReleaseLibrary)> library(raw, curtcReleaseLibrary);
            EXPECT_EQ(statistics.kernelCompilations, 1);
            EXPECT_GT(statistics.frontendNs, 0);
            EXPECT_LE(statistics.frontendNs, statistics.totalNs);
            CUkernel kernel{};
            ASSERT_EQ(curtcGetKernel(library.get(), to_view("kernel"), &kernel), CLANGRTC_SUCCESS);
            test::launch(kernel, 2 * value);

            curtcLibrary hit{};
            ASSERT_EQ(curtcGet(cache.get(), &request, diagnostic, &log, &hit, &statistics), CLANGRTC_SUCCESS) << log;
            EXPECT_EQ(statistics.kernelCompilations, 0);
            EXPECT_EQ(statistics.frontendNs, 0);
            curtcReleaseLibrary(hit);
        }

        auto invalid = std::string("extern \"C\" __global__ void kernel(int *out) { *out = missing_function(); }\n");
        input.source.source.data = to_view(invalid);
        curtcLibrary failed{};
        curtcCompileStatistics statistics{};
        EXPECT_EQ(curtcGet(cache.get(), &request, nullptr, nullptr, &failed, &statistics), CLANGRTC_COMPILATION_FAILED);
        EXPECT_EQ(failed, nullptr);
        EXPECT_GT(statistics.frontendNs, 0);
    }
}

/** @brief Frontend statistics sum actual kernel and UDF work while cached requests report no compiler activity. */
TEST(CurtcGpu, AggregatesFrontendStatisticsAcrossLinkedInputs)
{
    ASSERT_EQ(cudaSetDevice(test::GpuOrdinal), cudaSuccess);
    ASSERT_EQ(cudaFree(nullptr), cudaSuccess);
    auto target = test::gpu_target();
    auto preparation = test::request("using Prepared = int;\n", {});
    preparation.source.path = to_view("/__gpu__/environment.cuh");
    test::Options options{std::format("--cuda-path={}", test::CudaPath), std::format("--gpu-architecture={}", target),
                          "--artifact=pch"};
    auto prepared = test::prepare(preparation, options);
    ASSERT_EQ(prepared.status, CLANGRTC_SUCCESS) << prepared.log();
    auto kernelEnvironment = test::load(prepared.output());
    auto udfEnvironment = test::load(prepared.output());
    ASSERT_EQ(kernelEnvironment.status, CLANGRTC_SUCCESS);
    ASSERT_EQ(udfEnvironment.status, CLANGRTC_SUCCESS);
    curtcConfiguration configuration{};
    ASSERT_EQ(curtcDefaultConfiguration(&configuration), CLANGRTC_SUCCESS);
    configuration.diskCache = false;
    curtcCache rawCache{};
    ASSERT_EQ(curtcCreateCache(&configuration, &rawCache), CLANGRTC_SUCCESS);
    std::unique_ptr<curtcCacheImpl, decltype(&curtcDestroyCache)> cache(rawCache, curtcDestroyCache);
    curtcKernelSelection selection{to_view("&kernel"), {}};
    curtcInput inputs[2]{};
    inputs[0].type = CURTC_INPUT_TYPE_ARTIFACT;
    inputs[0].source.artifact = kernelEnvironment.artifact.get();
    inputs[0].entries = &selection;
    inputs[0].entryCount = 1;
    inputs[1].type = CURTC_INPUT_TYPE_ARTIFACT;
    inputs[1].source.artifact = udfEnvironment.artifact.get();
    curtcRequest request{};
    request.inputs = inputs;
    request.inputCount = std::size(inputs);
    request.target = to_view(target);

    for (int value : {3, 7})
    {
        auto kernelSource = std::format("extern \"C\" __device__ int udf(int);\n"
                                        "extern \"C\" __global__ void kernel(int *out) "
                                        "{{ *out = udf(int(threadIdx.x)) + {}; }}\n",
                                        value);
        auto udfSource = std::format("extern \"C\" __device__ int udf(int x) {{ return x + {}; }}\n", value);
        inputs[0].source.source = {to_view("/__gpu__/kernel.cu"), to_view(kernelSource)};
        inputs[1].source.source = {to_view("/__gpu__/udf.cu"), to_view(udfSource)};
        curtcLibrary raw{};
        curtcCompileStatistics statistics{};
        std::string log;
        auto diagnostic = [](clangrtcDiagnostic const *message, void *context) {
            static_cast<std::string *>(context)->append(message->message.data, message->message.size);
        };
        ASSERT_EQ(curtcGet(cache.get(), &request, diagnostic, &log, &raw, &statistics), CLANGRTC_SUCCESS) << log;
        std::unique_ptr<curtcLibraryImpl, decltype(&curtcReleaseLibrary)> library(raw, curtcReleaseLibrary);
        EXPECT_EQ(statistics.kernelCompilations, 1);
        EXPECT_EQ(statistics.fragmentCompilations, 1);
        EXPECT_GT(statistics.frontendNs, 0);
        EXPECT_LE(statistics.frontendNs, statistics.totalNs);
        CUkernel kernel{};
        ASSERT_EQ(curtcGetKernel(library.get(), to_view("kernel"), &kernel), CLANGRTC_SUCCESS);
        test::launch(kernel, 2 * value);
        curtcLibrary hit{};
        ASSERT_EQ(curtcGet(cache.get(), &request, diagnostic, &log, &hit, &statistics), CLANGRTC_SUCCESS) << log;
        EXPECT_EQ(statistics.frontendNs, 0);
        curtcReleaseLibrary(hit);
    }
}

/**
 * @brief Execute multiple kernels sharing a library with a runtime-linked device implementation.
 *
 * Checks GPU results as the implementation changes, repeated-request handle reuse, and ownership after cache
 * destruction.
 */
TEST(CurtcGpu, RunsMultipleKernelsAndRuntimeLinkedFunctions)
{
    ASSERT_EQ(cudaSetDevice(test::GpuOrdinal), cudaSuccess);
    ASSERT_EQ(cudaFree(nullptr), cudaSuccess);
    int major{}, minor{};
    ASSERT_EQ(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, test::GpuOrdinal), cudaSuccess);
    ASSERT_EQ(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, test::GpuOrdinal), cudaSuccess);
    std::string target = std::format("sm_{}{}", major, minor);
    curtcConfiguration config{};
    ASSERT_EQ(curtcDefaultConfiguration(&config), CLANGRTC_SUCCESS);
    curtcCache raw_cache{};
    ASSERT_EQ(curtcCreateCache(&config, &raw_cache), CLANGRTC_SUCCESS);
    std::unique_ptr<curtcCacheImpl, decltype(&curtcDestroyCache)> cache(raw_cache, curtcDestroyCache);
    std::string source = R"CUDA(
        extern "C" __global__ void fixed(int *out)
        {
            *out = 42;
        }

        extern "C" __device__ int udf(int);

        extern "C" __global__ void staged(int *out)
        {
            *out = udf(20);
        }
    )CUDA";
    std::string flags = std::format("--cuda-path={}", test::CudaPath);
    clangrtcString option = to_view(flags);
    curtcInput base{};
    base.source.source = {to_view("/__gpu__/kernels.cu"), to_view(source)};
    base.options = &option;
    base.optionCount = 1;
    curtcRequest request{};
    request.inputs = &base;
    request.inputCount = 1;
    curtcKernelSelection selected[]{{to_view("&fixed"), to_view("fixed")}, {to_view("&staged"), to_view("staged")}};
    base.entries = selected;
    base.entryCount = 2;
    request.target = to_view(target);

    for (int offset : {2, 3})
    {
        std::string udf = std::format("extern \"C\" __device__ int udf(int x) {{return x*2+{};}}", offset);
        curtcInput fragment{};
        fragment.source.source.data = to_view(udf);
        fragment.options = &option;
        fragment.optionCount = 1;
        curtcInput inputs[] = {base, fragment};
        request.inputs = inputs;
        request.inputCount = 2;
        curtcLibrary result{}, repeated{};
        ASSERT_EQ(curtcGet(cache.get(), &request, nullptr, nullptr, &result, nullptr), CLANGRTC_SUCCESS);
        ASSERT_EQ(curtcGet(cache.get(), &request, nullptr, nullptr, &repeated, nullptr), CLANGRTC_SUCCESS);
        curtcKernel kernels[2]{}, cached[2]{};
        size_t count = 2;
        ASSERT_EQ(curtcEnumerateKernels(result, &count, kernels), CLANGRTC_SUCCESS);
        ASSERT_EQ(count, 2);
        ASSERT_EQ(curtcEnumerateKernels(repeated, &count, cached), CLANGRTC_SUCCESS);
        cache.reset(); // The last result's owner keeps its CUlibrary alive independently.
        CUkernel handles[2]{}, cached_handles[2]{};

        for (size_t i = 0; i < count; ++i)
        {
            ASSERT_EQ(curtcGetKernel(result, kernels[i].symbolName, &handles[i]), CLANGRTC_SUCCESS);
            ASSERT_EQ(curtcGetKernel(repeated, cached[i].symbolName, &cached_handles[i]), CLANGRTC_SUCCESS);
            EXPECT_EQ(handles[i], cached_handles[i]);
        }

        test::launch(handles[0], 42);
        test::launch(handles[1], 40 + offset);
        curtcReleaseLibrary(repeated);
        curtcReleaseLibrary(result);

        if (offset == 2)
        {
            ASSERT_EQ(curtcCreateCache(&config, &raw_cache), CLANGRTC_SUCCESS);
            cache.reset(raw_cache);
        }
    }
}

/**
 * @brief Execute toolkit-provided vector factories and atomic overloads across device header modes.
 *
 * Covers floating-point/scoped atomics, 64-bit minimum, integer maximum/CAS, address-space predicates, warp shuffle,
 * and integer dot products on a real GPU. The caller must retain Clang's identity after the supplied CUDA headers are
 * imported.
 */
TEST(CurtcGpu, RunsToolkitDeviceHelpersWithNvrtcProfile)
{
    curtcConfiguration configuration{};
    ASSERT_EQ(curtcDefaultConfiguration(&configuration), CLANGRTC_SUCCESS);
    curtcCache raw_cache{};
    ASSERT_EQ(curtcCreateCache(&configuration, &raw_cache), CLANGRTC_SUCCESS);
    std::unique_ptr<curtcCacheImpl, decltype(&curtcDestroyCache)> cache(raw_cache, curtcDestroyCache);
    std::string source = R"CUDA(
#include <cassert>
#ifdef __CUDACC_RTC__
#error "Clang must not identify as NVRTC"
#endif

        extern "C" __global__ void toolkit_helpers(int *out)
        {
            __shared__ float sum;
            __shared__ double precise;
            __shared__ unsigned long long wide;
            sum = 1.0f;
            precise = 0.25;
            wide = 8;
            *out = 0;
            atomicAdd(&sum, 2.5f);
            atomicAdd(&precise, 0.75);
            atomicMin(&wide, 5ULL);
            auto values = make_int2(4, 6);
            atomicMax(out, values.x);
            atomicCAS(out, values.x, values.y);
            *out += int(sum) + int(wide) + __isGlobal(out) + __isShared(&sum);
            atomicAdd_block(out, 1);
            atomicAdd_system(out, 1);
            auto dot = __dp4a(0x01010101, 0x02020202, 0);
            *out += __shfl_sync(1u, dot, 0);
            assert(sum == 3.5f && precise == 1.0 && wide == 5 && *out == 26);
        }
    )CUDA";
    auto diagnostic = [](clangrtcDiagnostic const *value, void *context) {
        static_cast<std::string *>(context)->append(value->message.data, value->message.size);
    };

    for (bool libcudacxx : {false, true})
    {
        if (libcudacxx && !test::CcclFreestanding)
        {
            continue;
        }

        for (bool minimal : {false, true})
        {
            SCOPED_TRACE(std::format("libcudacxx={}, minimal={}", libcudacxx, minimal));
            test::Options options{std::format("--cuda-path={}", test::CudaPath), "--profile=nvrtc",
                                  "--clang-option=--cuda-feature=+ptx60"};

            if (minimal)
            {
                options.push_back("--minimal");
            }

            if (libcudacxx)
            {
                options.push_back("--stdlib=libcudacxx");
                options.push_back(std::format("-I{}", test::CcclPath));
            }

            auto flags = test::to_views(options);
            curtcInput input{};
            input.source.source = {to_view("/__gpu__/toolkit_helpers.cu"), to_view(source)};
            input.options = flags.data();
            input.optionCount = flags.size();
            curtcRequest request{};
            request.inputs = &input;
            request.inputCount = 1;
            request.target = to_view(test::gpu_target());
            curtcLibrary raw_library{};
            std::string log;
            ASSERT_EQ(curtcGet(cache.get(), &request, diagnostic, &log, &raw_library, nullptr), CLANGRTC_SUCCESS)
                << log;
            std::unique_ptr<curtcLibraryImpl, decltype(&curtcReleaseLibrary)> library(raw_library, curtcReleaseLibrary);
            CUkernel kernel{};
            ASSERT_EQ(curtcGetKernel(library.get(), to_view("toolkit_helpers"), &kernel), CLANGRTC_SUCCESS);
            test::launch(kernel, 26);
        }
    }
}

/**
 * @brief Link and execute nonconstant libdevice math and bit operations across all device header modes.
 *
 * Both independently compiled inputs use libdevice, exercising shared device dependencies during LTO linking.
 */
TEST(CurtcGpu, LibdeviceFunctionsExecuteAcrossHeaderModes)
{
    curtcConfiguration configuration{};
    ASSERT_EQ(curtcDefaultConfiguration(&configuration), CLANGRTC_SUCCESS);
    configuration.diskCache = false;
    curtcCache rawCache{};
    ASSERT_EQ(curtcCreateCache(&configuration, &rawCache), CLANGRTC_SUCCESS);
    std::unique_ptr<curtcCacheImpl, decltype(&curtcDestroyCache)> cache(rawCache, curtcDestroyCache);
    auto source = std::string_view(R"CUDA(
#include <cmath>

        extern "C" __device__ int math_fragment(unsigned);

        extern "C" __global__ void kernel(int *out)
        {
            double offset = double(threadIdx.x);
            *out = int(std::sqrt(25.0 + offset) + 0.5) + int(std::log2(8.0 + offset) + 0.5) + int(std::sin(offset)) +
                   __popc(7u + threadIdx.x) + math_fragment(threadIdx.x);
        }
    )CUDA");
    auto fragment = std::string_view(R"CUDA(
#include <cmath>

        extern "C" __device__ int math_fragment(unsigned offset)
        {
            return int(std::sqrt(25.0 + double(offset)) + 0.5) + __popc(7u + offset);
        }
    )CUDA");
    auto target = test::gpu_target();

    for (bool libcudacxx : {false, true})
    {
        if (libcudacxx && !test::CcclFreestanding)
        {
            continue;
        }

        for (bool minimal : {false, true})
        {
            SCOPED_TRACE(std::format("libcudacxx={}, minimal={}", libcudacxx, minimal));
            test::Options options{std::format("--cuda-path={}", test::CudaPath)};

            if (minimal)
            {
                options.push_back("--minimal");
            }

            if (libcudacxx)
            {
                options.push_back("--stdlib=libcudacxx");
                options.push_back(std::format("-I{}", test::CcclPath));
            }

            auto flags = test::to_views(options);
            curtcKernelSelection selection{to_view("&kernel"), {}};
            curtcInput input{};
            input.type = CURTC_INPUT_TYPE_CUDA;
            input.source.source = {to_view("/__gpu__/libdevice.cu"), to_view(source)};
            input.options = flags.data();
            input.optionCount = flags.size();
            input.entries = &selection;
            input.entryCount = 1;
            curtcInput udf{};
            udf.type = CURTC_INPUT_TYPE_CUDA;
            udf.source.source = {to_view("/__gpu__/libdevice-fragment.cu"), to_view(fragment)};
            udf.options = flags.data();
            udf.optionCount = flags.size();
            curtcInput inputs[]{input, udf};
            curtcRequest request{};
            request.inputs = inputs;
            request.inputCount = std::size(inputs);
            request.target = to_view(target);
            std::string log;
            auto diagnostic = [](clangrtcDiagnostic const *message, void *context) {
                static_cast<std::string *>(context)->append(message->message.data, message->message.size);
            };
            curtcLibrary rawLibrary{};
            ASSERT_EQ(curtcGet(cache.get(), &request, diagnostic, &log, &rawLibrary, nullptr), CLANGRTC_SUCCESS) << log;
            std::unique_ptr<curtcLibraryImpl, decltype(&curtcReleaseLibrary)> library(rawLibrary, curtcReleaseLibrary);
            CUkernel kernel{};
            ASSERT_EQ(curtcGetKernel(library.get(), to_view("kernel"), &kernel), CLANGRTC_SUCCESS);
            test::launch(kernel, 19);
        }
    }
}
} // namespace
