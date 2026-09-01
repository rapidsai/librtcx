/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <rtcx/cuda_library_compat.hpp>

#include <cuda_runtime.h>

#include <driver_types.h>
#include <vector_types.h>

#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>

namespace rtcx {

struct algorithm_launcher {
  algorithm_launcher() : kernel{nullptr}, library{nullptr} {}

  algorithm_launcher(cudaKernel_t k, cuda_library_t lib);

  ~algorithm_launcher();

  algorithm_launcher(algorithm_launcher const&)            = delete;
  algorithm_launcher& operator=(algorithm_launcher const&) = delete;

  algorithm_launcher(algorithm_launcher&& other) noexcept;
  algorithm_launcher& operator=(algorithm_launcher&& other) noexcept;

  template <typename FuncT, typename... Args>
  void dispatch(cudaStream_t stream, dim3 grid, dim3 block, std::size_t shared_mem, Args&&... args)
  {
    static_assert(std::is_same_v<FuncT, void(std::remove_reference_t<Args>...)>,
                  "dispatch() argument types do not match the kernel function signature FuncT");

    void* kernel_args[] = {const_cast<void*>(static_cast<void const*>(&args))...};
    this->call(stream, grid, block, shared_mem, kernel_args);
  }

  template <typename FuncT, typename... Args>
  void dispatch_cooperative(
    cudaStream_t stream, dim3 grid, dim3 block, std::size_t shared_mem, Args&&... args)
  {
    static_assert(
      std::is_same_v<FuncT, void(std::remove_reference_t<Args>...)>,
      "dispatch_cooperative() argument types do not match the kernel function signature FuncT");

    void* kernel_args[] = {const_cast<void*>(static_cast<void const*>(&args))...};
    this->call_cooperative(stream, grid, block, shared_mem, kernel_args);
  }

  cudaKernel_t get_kernel() { return this->kernel; }

 private:
  void call(cudaStream_t stream, dim3 grid, dim3 block, std::size_t shared_mem, void** args);
  void call_cooperative(
    cudaStream_t stream, dim3 grid, dim3 block, std::size_t shared_mem, void** args);
  cudaKernel_t kernel;
  cuda_library_t library;
};

}  // namespace rtcx
