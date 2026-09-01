/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cuda_runtime.h"
#include "nvJitLink.h"

#include <rtcx/algorithm_planner.hpp>
#include <rtcx/macros.hpp>
#include <rtcx/nvjitlink_checker.hpp>

#include <chrono>
#include <iterator>
#include <memory>
#include <mutex>
#include <new>
#include <shared_mutex>
#include <string>
#include <vector>

namespace rtcx {

std::string algorithm_planner::get_fragments_key() const
{
  std::string key = "";
  for (auto const& fragment : this->fragments) {
    key += fragment->get_key();
  }
  return key;
}

std::shared_ptr<algorithm_launcher> algorithm_planner::read_cache(
  std::string const& launch_key) const
{
  auto& launchers = jit_cache_.launchers;
  std::shared_lock<std::shared_mutex> read_lock(jit_cache_.mutex);
  if (auto it = launchers.find(launch_key); it != launchers.end()) { return it->second; }
  return nullptr;
}

std::shared_ptr<algorithm_launcher> algorithm_planner::get_launcher()
{
  auto& launchers = jit_cache_.launchers;
  auto launch_key = this->get_fragments_key();

  if (auto hit = read_cache(launch_key)) { return hit; }

  std::unique_lock<std::shared_mutex> write_lock(jit_cache_.mutex);
  if (auto it = launchers.find(launch_key); it != launchers.end()) { return it->second; }

  auto launcher         = this->build();
  launchers[launch_key] = launcher;
  return launcher;
}

std::shared_ptr<algorithm_launcher> algorithm_planner::build()
{
  int device = 0;
  int major  = 0;
  int minor  = 0;
  RTCX_CUDA_TRY(cudaGetDevice(&device));
  RTCX_CUDA_TRY(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
  RTCX_CUDA_TRY(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));

  std::string archs = "-arch=sm_" + std::to_string((major * 10 + minor));

  // Load the generated LTO IR and link them together
  nvJitLinkHandle handle;
  std::vector<char const*> lopts;
  lopts.reserve(2 + linktime_extra_options.size());
  lopts.push_back("-lto");
  lopts.push_back(archs.c_str());
  for (auto const& opt : linktime_extra_options) {
    lopts.push_back(opt.c_str());
  }
  auto result = nvJitLinkCreate(&handle, static_cast<unsigned int>(lopts.size()), lopts.data());
  check_nvjitlink_result(handle, result);

  for (auto const& frag : this->fragments) {
    frag->add_to(handle);
  }

  // Call to nvJitLinkComplete causes linker to link together all the LTO-IR
  // modules perform any optimizations and generate cubin from it.
  result = nvJitLinkComplete(handle);
  check_nvjitlink_result(handle, result);

  // get cubin from nvJitLink
  size_t cubin_size;
  result = nvJitLinkGetLinkedCubinSize(handle, &cubin_size);
  check_nvjitlink_result(handle, result);

  std::unique_ptr<char[]> cubin{new char[cubin_size]};
  result = nvJitLinkGetLinkedCubin(handle, cubin.get());
  check_nvjitlink_result(handle, result);

  result = nvJitLinkDestroy(&handle);
  RTCX_EXPECTS(result == NVJITLINK_SUCCESS, "nvJitLinkDestroy failed");

  // cubin is linked, so now load it
  cuda_library_t library;
  RTCX_CUDA_TRY(library_load_data(&library, cubin.get()));

  cudaKernel_t kernel;
  RTCX_CUDA_TRY(library_get_kernel(&kernel, library, this->entrypoint.c_str()));

  return std::make_shared<algorithm_launcher>(kernel, library);
}

}  // namespace rtcx
