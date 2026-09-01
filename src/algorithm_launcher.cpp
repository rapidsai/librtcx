/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rtcx/algorithm_launcher.hpp>
#include <rtcx/macros.hpp>

namespace rtcx {

algorithm_launcher::algorithm_launcher(cudaKernel_t k, cuda_library_t lib) : kernel{k}, library{lib}
{
}

algorithm_launcher::~algorithm_launcher()
{
  if (library != nullptr) { (void)library_unload(library); }
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
    if (library != nullptr) { (void)library_unload(library); }
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
  RTCX_CUDA_TRY(
    launch_kernel(kernel, grid, block, shared_mem, stream, kernel_args, /* cooperative = */ false));
}

void algorithm_launcher::call_cooperative(
  cudaStream_t stream, dim3 grid, dim3 block, std::size_t shared_mem, void** kernel_args)
{
  RTCX_CUDA_TRY(
    launch_kernel(kernel, grid, block, shared_mem, stream, kernel_args, /* cooperative = */ true));
}

}  // namespace rtcx
