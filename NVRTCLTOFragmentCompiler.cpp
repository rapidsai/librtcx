/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cuda.h"
#include "macros.hpp"

#include <NVRTCLTOFragmentCompiler.hpp>
#include <nvrtc.h>

#include <mutex>

#define NVRTC_SAFE_CALL(_call)                                                 \
  {                                                                            \
    nvrtcResult result = _call;                                                \
    std::string error_string =                                                 \
      std::string("nvrtc error: ") + std::string(nvrtcGetErrorString(result)); \
    RTCX_EXPECTS(result == NVRTC_SUCCESS, "%s", error_string.c_str());         \
  }

NVRTCLTOFragmentCompiler::NVRTCLTOFragmentCompiler()
{
  int device = 0;
  int major  = 0;
  int minor  = 0;
  RTCX_CUDA_TRY(cudaGetDevice(&device));
  RTCX_CUDA_TRY(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
  RTCX_CUDA_TRY(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));

  this->standard_compile_opts = {
    std::string{"-arch=sm_" + std::to_string((major * 10 + minor))},
    std::string{"-dlto"},
    std::string{"-rdc=true"},
    std::string{"--std=c++20"},
    std::string{"-default-device"},
  };
}

std::unique_ptr<UDFFatbinFragment> NVRTCLTOFragmentCompiler::read_cache(
  std::string const& key) const
{
  std::shared_lock<std::shared_mutex> read_lock(cache_mutex_);
  if (auto it = cache.find(key); it != cache.end()) {
    return std::make_unique<UDFFatbinFragment>(key, it->second);
  }
  return nullptr;
}

std::unique_ptr<UDFFatbinFragment> NVRTCLTOFragmentCompiler::compile(std::string const& key,
                                                                     std::string const& code)
{
  if (auto hit = read_cache(key)) { return hit; }

  std::unique_lock<std::shared_mutex> write_lock(cache_mutex_);
  if (auto it = cache.find(key); it != cache.end()) {
    return std::make_unique<UDFFatbinFragment>(key, it->second);
  }

  nvrtcProgram prog;
  NVRTC_SAFE_CALL(
    nvrtcCreateProgram(&prog, code.c_str(), "nvrtc_lto_fragment", 0, nullptr, nullptr));

  // Convert std::vector<std::string> to std::vector<const char*> for nvrtc API
  std::vector<char const*> opts;
  opts.reserve(this->standard_compile_opts.size());
  for (auto const& opt : this->standard_compile_opts) {
    opts.push_back(opt.c_str());
  }

  nvrtcResult compileResult = nvrtcCompileProgram(prog,          // prog
                                                  opts.size(),   // numOptions
                                                  opts.data());  // options

  try {
    if (compileResult != NVRTC_SUCCESS) {
      // Obtain compilation log from the program.
      size_t log_size;
      NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &log_size));
      std::unique_ptr<char[]> log{new char[log_size]};
      NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, log.get()));
      RTCX_FAIL("nvrtc compile error log: \n%s", log.get());
    }
  } catch (...) {
    NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));
    throw;
  }

  // Obtain generated LTO IR from the program.
  std::size_t ltoIRSize;
  NVRTC_SAFE_CALL(nvrtcGetLTOIRSize(prog, &ltoIRSize));

  std::vector<uint8_t> lto_ir(ltoIRSize);
  NVRTC_SAFE_CALL(nvrtcGetLTOIR(prog, reinterpret_cast<char*>(lto_ir.data())));

  NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));

  cache[key] = std::move(lto_ir);
  return std::make_unique<UDFFatbinFragment>(key, cache[key]);
}

NVRTCLTOFragmentCompiler& nvrtc_compiler()
{
  static NVRTCLTOFragmentCompiler compiler;
  return compiler;
}
