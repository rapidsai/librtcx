/* SPDX-License-Identifier: Apache-2.0 */

/**
 * @file gpu_support.cpp
 * @brief Compile valid CUDA images and execute kernels on the configured physical GPU.
 *
 * Tests fail when CUDA is unavailable. No replacement driver or synthetic executable image is used.
 */
#include "gpu_support.hpp"

#include "error.hpp"
#include "support.hpp"
#include "test_config.hpp"

#include <cuda_runtime_api.h>
#include <format>
#include <gtest/gtest.h>
#include <memory>
#include <nvJitLink.h>

namespace test
{
namespace
{
/** @brief Establish the selected device's real primary context before running GPU tests. */
class GpuEnvironment : public ::testing::Environment
{
  public:
    /** @brief Fail the GPU suite if the actual CUDA runtime cannot initialize the selected device. */
    void SetUp() override
    {
        EXPECT_EQ(cudaSetDevice(GpuOrdinal), cudaSuccess);
        EXPECT_EQ(cudaFree(nullptr), cudaSuccess);
    }
};

/** @brief Register GPU initialization without touching CUDA during GoogleTest discovery. */
[[maybe_unused]] auto *environment = ::testing::AddGlobalTestEnvironment(new GpuEnvironment);

/** @brief Preserve the real linker diagnostic when fixture generation fails. */
void check_link(nvJitLinkResult status, nvJitLinkHandle handle)
{
    if (status == NVJITLINK_SUCCESS)
    {
        return;
    }

    size_t size{};
    nvJitLinkGetErrorLogSize(handle, &size);
    std::string log(size, '\0');

    if (size)
    {
        nvJitLinkGetErrorLog(handle, log.data());
    }

    throw std::runtime_error(std::format("CUDA fixture linking failed ({}): {}", static_cast<int>(status), log));
}

/** @brief Execute a scalar-output kernel while keeping its allocation alive until synchronization. */
template <class T> void launch_scalar(CUkernel kernel, T expected)
{
    ASSERT_EQ(cudaSetDevice(GpuOrdinal), cudaSuccess);
    CUfunction function{};
    ASSERT_EQ(cuKernelGetFunction(&function, kernel), CUDA_SUCCESS);
    T *pointer{};
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&pointer), sizeof(T)), cudaSuccess);
    std::unique_ptr<T, decltype(&cudaFree)> allocation(pointer, cudaFree);
    ASSERT_EQ(cudaMemset(pointer, 0, sizeof(T)), cudaSuccess);
    void *arguments[]{&pointer};
    ASSERT_EQ(cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, nullptr, arguments, nullptr), CUDA_SUCCESS);
    T actual{};
    ASSERT_EQ(cudaMemcpy(&actual, pointer, sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(actual, expected);
}
} // namespace

std::string const &gpu_target()
{
    static std::string const target = [] {
        cudaDeviceProp properties{};
        auto status = cudaGetDeviceProperties(&properties, GpuOrdinal);

        if (status != cudaSuccess)
        {
            throw std::runtime_error(
                std::format("Cannot query CUDA GPU {}: {}", GpuOrdinal, cudaGetErrorString(status)));
        }

        return std::format("sm_{}{}", properties.major, properties.minor);
    }();
    return target;
}

std::vector<char> cubin_fixture(int value)
{
    auto text = std::format("#define VALUE {}\n{}", value, R"CUDA(
        extern "C" __global__ void first(int *out)
        {
            *out = VALUE;
        }

        extern "C" __global__ void second(int *out)
        {
            *out = VALUE;
        }

        extern "C" __global__ void fixed(int *out)
        {
            *out = VALUE;
        }

        extern "C" __global__ void c_kernel(int *out)
        {
            *out = VALUE;
        }
    )CUDA");
    auto result = compile(request(text, {}), {std::format("--cuda-path={}", CudaPath),
                                              std::format("--gpu-architecture={}", gpu_target()), "--lto-ir"});

    if (result.status != CLANGRTC_SUCCESS)
    {
        throw std::runtime_error(result.log());
    }

    auto code = result.output();
    auto target = std::format("-arch={}", gpu_target());
    char const *options[]{target.c_str(), "-lto", "-nvvm-latest", "-no-cache"};
    nvJitLinkHandle linker{};
    CR_DEFER([&] {
        if (linker)
        {
            nvJitLinkDestroy(&linker);
        }
    });
    check_link(nvJitLinkCreate(&linker, 4, options), linker);
    check_link(nvJitLinkAddData(linker, NVJITLINK_INPUT_LTOIR, code.data(), code.size(), "fixture"), linker);
    check_link(nvJitLinkComplete(linker), linker);
    size_t size{};
    check_link(nvJitLinkGetLinkedCubinSize(linker, &size), linker);
    std::vector<char> cubin(size);
    check_link(nvJitLinkGetLinkedCubin(linker, cubin.data()), linker);
    return cubin;
}

std::vector<char> cubin_fixture()
{
    return cubin_fixture(42);
}

void launch(CUkernel kernel, int expected)
{
    launch_scalar(kernel, expected);
}

void launch(CUkernel kernel, float expected)
{
    launch_scalar(kernel, expected);
}
} // namespace test
