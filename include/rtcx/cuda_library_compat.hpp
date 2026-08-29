/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file cuda_library_compat.hpp
 * @brief Toolkit-version shim for the CUDA library-management API.
 *
 * rtcx links LTO-IR fragments into a cubin at run time, loads that image, looks up an entry point
 * in it and launches it. The CUDA *runtime* API for this -- `cudaLibrary_t`, `cudaLibraryLoadData`,
 * `cudaLibraryGetKernel`, `cudaLibraryUnload`, and passing the resulting `cudaKernel_t` to
 * `cudaLaunchKernelExC` -- only exists from CUDA 12.8 onwards. It is not merely hidden behind a
 * header guard on older toolkits: the entry points are absent from `libcudart` as well, so nothing
 * a caller defines can bring them back.
 *
 * The equivalent *driver* API (`cuLibraryLoadData`, `cuLibraryGetKernel`, `cuLibraryUnload`,
 * `cuLaunchKernelEx`) has been available since CUDA 12.0, and the handles are the same objects:
 * `cudaLibrary_t` is `struct CUlib_st*`, i.e. `CUlibrary`, and `cudaKernel_t` is
 * `struct CUkern_st*`, i.e. `CUkernel`. This header forwards to the runtime API when it is
 * available and to the driver API otherwise, which lets rtcx -- and therefore its consumers -- be
 * built with a CUDA 12.0-12.6 toolkit.
 *
 * The shim is inert on 12.8+: everything below compiles down to the original runtime calls and no
 * driver dependency is added.
 */

#pragma once

#include <cuda_runtime.h>

#include <cstddef>

/** 1 when the CUDA runtime provides the library-management API natively, 0 when it is emulated. */
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 12080)
#define RTCX_HAS_RUNTIME_LIBRARY_API 1
#else
#define RTCX_HAS_RUNTIME_LIBRARY_API 0
#endif

#if !RTCX_HAS_RUNTIME_LIBRARY_API
#include <cuda.h>
#endif

namespace rtcx {

#if RTCX_HAS_RUNTIME_LIBRARY_API
using library_t = cudaLibrary_t;
#else
using library_t = CUlibrary;
#endif

namespace detail {

#if !RTCX_HAS_RUNTIME_LIBRARY_API

/** Translate a driver-API status into the closest runtime-API status. */
inline cudaError_t from_driver(CUresult status)
{
  switch (status) {
    case CUDA_SUCCESS: return cudaSuccess;
    case CUDA_ERROR_OUT_OF_MEMORY: return cudaErrorMemoryAllocation;
    case CUDA_ERROR_NOT_INITIALIZED: return cudaErrorInitializationError;
    case CUDA_ERROR_DEINITIALIZED: return cudaErrorCudartUnloading;
    case CUDA_ERROR_NO_DEVICE: return cudaErrorNoDevice;
    case CUDA_ERROR_INVALID_DEVICE: return cudaErrorInvalidDevice;
    case CUDA_ERROR_INVALID_VALUE: return cudaErrorInvalidValue;
    case CUDA_ERROR_INVALID_HANDLE: return cudaErrorInvalidResourceHandle;
    case CUDA_ERROR_NOT_FOUND: return cudaErrorSymbolNotFound;
    case CUDA_ERROR_INVALID_IMAGE: return cudaErrorInvalidKernelImage;
    case CUDA_ERROR_INVALID_PTX: return cudaErrorInvalidPtx;
    case CUDA_ERROR_NO_BINARY_FOR_GPU: return cudaErrorNoKernelImageForDevice;
    case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES: return cudaErrorLaunchOutOfResources;
    case CUDA_ERROR_LAUNCH_FAILED: return cudaErrorLaunchFailure;
    case CUDA_ERROR_INVALID_CONTEXT: return cudaErrorDeviceUninitialized;
    default: return cudaErrorUnknown;
  }
}

/**
 * Make sure the runtime's primary context exists and is current.
 *
 * Driver-API calls such as `cuKernelGetFunction` operate on the current context, while the CUDA
 * runtime creates and binds its primary context lazily. `cudaFree(nullptr)` is the documented,
 * cheap and idempotent way to force that to have happened; without it an otherwise correct
 * sequence fails with `CUDA_ERROR_INVALID_CONTEXT`.
 */
inline cudaError_t ensure_context() { return cudaFree(nullptr); }

#endif  // !RTCX_HAS_RUNTIME_LIBRARY_API

}  // namespace detail

/** Load a linked cubin (or fatbin) image and return a library handle. */
inline cudaError_t library_load_data(library_t* library, void const* image)
{
#if RTCX_HAS_RUNTIME_LIBRARY_API
  return cudaLibraryLoadData(library, image, nullptr, nullptr, 0, nullptr, nullptr, 0);
#else
  auto status = detail::ensure_context();
  if (status != cudaSuccess) { return status; }
  return detail::from_driver(
    cuLibraryLoadData(library, image, nullptr, nullptr, 0, nullptr, nullptr, 0));
#endif
}

/** Look up an entry point by name in a loaded library. */
inline cudaError_t library_get_kernel(cudaKernel_t* kernel, library_t library, char const* name)
{
#if RTCX_HAS_RUNTIME_LIBRARY_API
  return cudaLibraryGetKernel(kernel, library, name);
#else
  return detail::from_driver(
    cuLibraryGetKernel(reinterpret_cast<CUkernel*>(kernel), library, name));
#endif
}

/** Unload a library previously returned by @ref library_load_data. */
inline cudaError_t library_unload(library_t library)
{
#if RTCX_HAS_RUNTIME_LIBRARY_API
  return cudaLibraryUnload(library);
#else
  return detail::from_driver(cuLibraryUnload(library));
#endif
}

/**
 * Launch a library kernel.
 *
 * @param kernel      entry point obtained from @ref library_get_kernel
 * @param grid        grid dimensions
 * @param block       block dimensions
 * @param shared_mem  dynamic shared memory, in bytes
 * @param stream      stream to launch on
 * @param args        array of pointers to the kernel arguments
 * @param cooperative whether to request a cooperative launch
 */
inline cudaError_t launch_kernel(cudaKernel_t kernel,
                                 dim3 grid,
                                 dim3 block,
                                 std::size_t shared_mem,
                                 cudaStream_t stream,
                                 void** args,
                                 bool cooperative)
{
#if RTCX_HAS_RUNTIME_LIBRARY_API
  cudaLaunchAttribute attribute[1];
  cudaLaunchConfig_t config{};
  config.gridDim          = grid;
  config.blockDim         = block;
  config.stream           = stream;
  config.dynamicSmemBytes = shared_mem;
  config.numAttrs         = 0;
  config.attrs            = nullptr;
  if (cooperative) {
    attribute[0].id              = cudaLaunchAttributeCooperative;
    attribute[0].val.cooperative = 1;
    config.numAttrs              = 1;
    config.attrs                 = attribute;
  }
  return cudaLaunchKernelExC(&config, kernel, args);
#else
  auto status = detail::ensure_context();
  if (status != cudaSuccess) { return status; }

  // `cuLaunchKernelEx` takes a CUfunction, so resolve the kernel against the current context.
  CUfunction function{};
  auto result = cuKernelGetFunction(&function, reinterpret_cast<CUkernel>(kernel));
  if (result != CUDA_SUCCESS) { return detail::from_driver(result); }

  CUlaunchAttribute attribute[1];
  CUlaunchConfig config{};
  config.gridDimX       = grid.x;
  config.gridDimY       = grid.y;
  config.gridDimZ       = grid.z;
  config.blockDimX      = block.x;
  config.blockDimY      = block.y;
  config.blockDimZ      = block.z;
  config.sharedMemBytes = static_cast<unsigned int>(shared_mem);
  config.hStream        = reinterpret_cast<CUstream>(stream);
  config.numAttrs       = 0;
  config.attrs          = nullptr;
  if (cooperative) {
    attribute[0].id                = CU_LAUNCH_ATTRIBUTE_COOPERATIVE;
    attribute[0].value.cooperative = 1;
    config.numAttrs                = 1;
    config.attrs                   = attribute;
  }
  return detail::from_driver(cuLaunchKernelEx(&config, function, args, nullptr));
#endif
}

}  // namespace rtcx
