/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime.h>

#include <cstddef>

/** 1 when the CUDA runtime provides the library-management API natively, 0 when it is emulated. */
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 12080)
#define RTCX_HAS_RUNTIME_LIBRARY_API 1
#else
#define RTCX_HAS_RUNTIME_LIBRARY_API 0
#include <cuda.h>
#endif

namespace rtcx {

/** Handle to a loaded library image. */
#if RTCX_HAS_RUNTIME_LIBRARY_API
using cuda_library_t = cudaLibrary_t;
#else
using cuda_library_t = CUlibrary;
#endif

/**
 * @brief Load a linked cubin (or fatbin) image.
 *
 * @param library Filled with a handle to the loaded image
 * @param image   Image to load
 * @return `cudaSuccess`, or the error that prevented the load
 */
cudaError_t library_load_data(cuda_library_t* library, void const* image);

/**
 * @brief Look up an entry point by name in a loaded library.
 *
 * @param kernel  Filled with a handle to the entry point
 * @param library Library to search
 * @param name    Mangled name of the entry point
 * @return `cudaSuccess`, or the error that prevented the lookup
 */
cudaError_t library_get_kernel(cudaKernel_t* kernel, cuda_library_t library, char const* name);

/**
 * @brief Unload a library loaded by @ref library_load_data.
 *
 * @param library Library to unload
 * @return `cudaSuccess`, or the error that prevented the unload
 */
cudaError_t library_unload(cuda_library_t library);

/**
 * @brief Launch a library kernel.
 *
 * @param kernel      Entry point obtained from @ref library_get_kernel
 * @param grid        Grid dimensions
 * @param block       Block dimensions
 * @param shared_mem  Dynamic shared memory, in bytes
 * @param stream      Stream to launch on
 * @param args        Array of pointers to the kernel arguments
 * @param cooperative Whether to request a cooperative launch
 * @return `cudaSuccess`, or the error that prevented the launch
 */
cudaError_t launch_kernel(cudaKernel_t kernel,
                          dim3 grid,
                          dim3 block,
                          std::size_t shared_mem,
                          cudaStream_t stream,
                          void** args,
                          bool cooperative);

}  // namespace rtcx
