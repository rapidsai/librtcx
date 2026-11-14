/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "rtcx_test_embed.hpp"

#include <cuda_runtime_api.h>

#include <gtest/gtest.h>
#include <rtcx/rtcx.hpp>

#include <cstdint>
#include <filesystem>
#include <format>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace {

#define CUDA_CHECK(call) ASSERT_EQ((call), cudaSuccess)

constexpr char write_value_source[] = R"cuda(
extern "C" __global__ void write_value(int* out) { *out = 42; }
)cuda";

std::string device_arch(char const* prefix)
{
  int device{};
  int major{};
  int minor{};
  EXPECT_EQ(cudaGetDevice(&device), cudaSuccess);
  EXPECT_EQ(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device), cudaSuccess);
  EXPECT_EQ(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device), cudaSuccess);
  return std::format("{}{}{}", prefix, major, minor);
}

void launch_and_read(rtcx::library const& library, char const* name, int& result)
{
  int* d_out{};
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_out), sizeof(int)));
  CUDA_CHECK(cudaMemset(d_out, 0, sizeof(int)));
  void* params[] = {&d_out};
  library->get_kernel(name).launch({1, 1, 1}, {1, 1, 1}, 0, nullptr, params);
  CUDA_CHECK(cudaMemcpy(&result, d_out, sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_out));
}

rtcx::byte_buffer compile_cubin(char const* source)
{
  auto arch                   = device_arch("--gpu-architecture=sm_");
  char const* const options[] = {arch.c_str()};
  return rtcx::compile({
    .name        = "write_value.cu",
    .source      = source,
    .options     = options,
    .target_type = rtcx::binary_type::CUBIN,
  });
}

class RtcxTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite()
  {
    // Make the primary context current for rtcx's driver API calls.
    ASSERT_EQ(cudaFree(nullptr), cudaSuccess);
    rtcx::initialize();
  }
};

TEST_F(RtcxTest, LibraryVersions)
{
  EXPECT_GE(rtcx::nvrtc_version(), 12000);
  EXPECT_GE(rtcx::nvjitlink_version(), 12000);
}

TEST_F(RtcxTest, CompileAndLaunch)
{
  auto cubin   = compile_cubin(write_value_source);
  auto library = rtcx::load_library({cubin.data(), cubin.size()});
  int result{};
  launch_and_read(library, "write_value", result);
  EXPECT_EQ(result, 42);
}

TEST_F(RtcxTest, CompileErrorThrows)
{
  EXPECT_ANY_THROW(std::ignore = compile_cubin("this is not CUDA"));
}

TEST_F(RtcxTest, LinkLtoIr)
{
  constexpr char kernel_source[] = R"cuda(
extern __device__ int udf(int);
extern "C" __global__ void write_value(int* out) { *out = udf(20); }
)cuda";
  constexpr char udf_source[]    = R"cuda(
__device__ int udf(int x) { return 2 * x + 2; }
)cuda";

  auto arch                           = device_arch("-arch=sm_");
  char const* const compile_options[] = {"-dlto", "-rdc=true", arch.c_str()};
  auto compile_lto                    = [&](char const* name, char const* source) {
    return rtcx::compile({
                         .name        = name,
                         .source      = source,
                         .options     = compile_options,
                         .target_type = rtcx::binary_type::LTO_IR,
    });
  };
  auto kernel_ir = compile_lto("kernel.cu", kernel_source);
  auto udf_ir    = compile_lto("udf.cu", udf_source);

  rtcx::memory_fragment const fragments[] = {
    {.data = {kernel_ir.data(), kernel_ir.size()},
     .type = rtcx::binary_type::LTO_IR,
     .name = "kernel.ltoir"},
    {.data = {udf_ir.data(), udf_ir.size()},
     .type = rtcx::binary_type::LTO_IR,
     .name = "udf.ltoir"},
  };
  char const* const link_options[] = {"-lto", arch.c_str()};
  auto cubin                       = rtcx::link_library({
                          .name             = "linked",
                          .output_type      = rtcx::binary_type::CUBIN,
                          .memory_fragments = fragments,
                          .link_options     = link_options,
  });

  auto library = rtcx::load_library({cubin.data(), cubin.size()});
  int result{};
  launch_and_read(library, "write_value", result);
  EXPECT_EQ(result, 42);
}

TEST_F(RtcxTest, CacheMemoryAndDisk)
{
  auto const root = std::filesystem::path{::testing::TempDir()} / "rtcx-cache-test";
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root / "objects");
  std::filesystem::create_directories(root / "staging");

  int compile_count = 0;
  auto compile      = [&]() -> std::tuple<rtcx::library, rtcx::blob> {
    ++compile_count;
    auto binary =
      std::make_shared<rtcx::blob_t>(rtcx::blob_t::from_buffer(compile_cubin(write_value_source)));
    return {rtcx::load_library(binary->view()), binary};
  };
  rtcx::hash128 const key{0x0123456789abcdefULL, 0xfedcba9876543210ULL};

  {
    rtcx::cache_t cache{(root / "objects").string(),
                        (root / "staging").string(),
                        rtcx::cache_limits{},
                        /* preload = */ false,
                        /* disable = */ false};
    auto first =
      cache.get_or_add_library(key, rtcx::library_compile_func::from_functor(compile)).get();
    auto second =
      cache.get_or_add_library(key, rtcx::library_compile_func::from_functor(compile)).get();
    EXPECT_EQ(first, second);
    EXPECT_EQ(compile_count, 1);
    EXPECT_EQ(cache.get_stats().library_mem_hits, 1);
  }

  // A new cache over the same directory loads the library from disk without recompiling.
  rtcx::cache_t cache{(root / "objects").string(),
                      (root / "staging").string(),
                      rtcx::cache_limits{},
                      /* preload = */ false,
                      /* disable = */ false};
  auto library =
    cache.get_or_add_library(key, rtcx::library_compile_func::from_functor(compile)).get();
  EXPECT_EQ(compile_count, 1);
  EXPECT_EQ(cache.get_stats().library_disk_hits, 1);

  int result{};
  launch_and_read(library, "write_value", result);
  EXPECT_EQ(result, 42);
  std::filesystem::remove_all(root);
}

TEST_F(RtcxTest, CompileWithEmbeddedHeaders)
{
  std::vector<char const*> headers;
  for (auto const& range : rtcx_test_embed::file_ranges) {
    headers.push_back(reinterpret_cast<char const*>(rtcx_test_embed::files.data() + range[0]));
  }

  auto arch = device_arch("--gpu-architecture=sm_");
  // NVRTC matches in-memory headers by their exact include name, so use the embed destination.
  char const* const options[] = {arch.c_str()};
  auto cubin                  = rtcx::compile({
                     .name                 = "embedded.cu",
                     .source               = R"cuda(
#include "include/rtcx_test/value.cuh"
extern "C" __global__ void write_value(int* out) { *out = rtcx_test::value(); }
)cuda",
                     .header_include_names = rtcx_test_embed::file_destinations,
                     .headers              = headers,
                     .options              = options,
                     .target_type          = rtcx::binary_type::CUBIN,
  });

  auto library = rtcx::load_library({cubin.data(), cubin.size()});
  int result{};
  launch_and_read(library, "write_value", result);
  EXPECT_EQ(result, 42);
}

}  // namespace
