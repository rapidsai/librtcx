/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rtcx/algorithm_launcher.hpp>
#include <rtcx/macros.hpp>

namespace rtcx {

algorithm_launcher::algorithm_launcher(cudaKernel_t k, cudaLibrary_t lib) : kernel{k}, library{lib}
{
}

algorithm_launcher::~algorithm_launcher()
{
  if (library != nullptr) { (void)cudaLibraryUnload(library); }
}

algorithm_launcher::algorithm_launcher(algorithm_launcher&& other) noexcept
  : kernel{other.kernel}, library{other.library}
{
  other.kernel  = nullptr;
  other.library = nullptr;
}

algorithm_launcher& algorithm_launcher::operator=(algorithm_launcher&& other) noexcept
{
  if (this != &other) {
    if (library != nullptr) { cudaLibraryUnload(library); }
    kernel        = other.kernel;
    library       = other.library;
    other.kernel  = nullptr;
    other.library = nullptr;
  }
  return *this;
}

void algorithm_launcher::call(
  cudaStream_t stream, dim3 grid, dim3 block, std::size_t shared_mem, void** kernel_args)
{
  cudaLaunchConfig_t config{};
  config.gridDim          = grid;
  config.blockDim         = block;
  config.stream           = stream;
  config.dynamicSmemBytes = shared_mem;
  config.numAttrs         = 0;
  config.attrs            = NULL;

  RTCX_CUDA_TRY(cudaLaunchKernelExC(&config, kernel, kernel_args));
}

void algorithm_launcher::call_cooperative(
  cudaStream_t stream, dim3 grid, dim3 block, std::size_t shared_mem, void** kernel_args)
{
  cudaLaunchAttribute attribute[1];
  attribute[0].id              = cudaLaunchAttributeCooperative;
  attribute[0].val.cooperative = 1;

  cudaLaunchConfig_t config{};
  config.gridDim          = grid;
  config.blockDim         = block;
  config.stream           = stream;
  config.dynamicSmemBytes = shared_mem;
  config.numAttrs         = 1;
  config.attrs            = attribute;

  RTCX_CUDA_TRY(cudaLaunchKernelExC(&config, kernel, kernel_args));
}

}  // namespace rtcx
