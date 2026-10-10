/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rtcx/cuda_library_compat.hpp>

#if !RTCX_HAS_RUNTIME_LIBRARY_API
#include <dlfcn.h>
#endif

namespace rtcx {

#if !RTCX_HAS_RUNTIME_LIBRARY_API

namespace {

#define RTCX_FOR_EACH_DRIVER_FUNC(DO_IT) \
  DO_IT(LibraryLoadData)                 \
  DO_IT(LibraryGetKernel)                \
  DO_IT(LibraryUnload)                   \
  DO_IT(KernelGetFunction)               \
  DO_IT(LaunchKernelEx)

// `RTCX_SYMBOL_NAME` expands its argument before stringifying it, so that the name looked up is
// the one `cuda.h` declares: under CUDA_API_PER_THREAD_DEFAULT_STREAM some of these are macros
// for a `_ptsz` entry point.
#define RTCX_STRINGIFY(symbol_) #symbol_
#define RTCX_SYMBOL_NAME(symbol_) RTCX_STRINGIFY(symbol_)

template <typename FuncT>
bool load_symbol(void* handle, char const* name, FuncT*& entry_point)
{
  entry_point = reinterpret_cast<FuncT*>(::dlsym(handle, name));
  return entry_point != nullptr;
}

// The driver entry points, resolved with dlopen and dlsym so that libcuda is not a link-time
// dependency and rtcx stays loadable where there is no driver. `loaded` is false when libcuda, or
// any one of the entry points, is unavailable.
struct driver_functions {
#define DO_IT(func_) decltype(::cu##func_)* func_ = nullptr;
  RTCX_FOR_EACH_DRIVER_FUNC(DO_IT)
#undef DO_IT

  bool loaded = false;

  driver_functions()
  {
    // Never dlclose()d: the libraries and kernels reached through these entry points stay valid
    // for as long as the process lives.
    void* handle = ::dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) { return; }

#define DO_IT(func_)                                                              \
  if (!load_symbol(handle, RTCX_SYMBOL_NAME(cu##func_), this->func_)) { return; }
    RTCX_FOR_EACH_DRIVER_FUNC(DO_IT)
#undef DO_IT

    loaded = true;
  }
};

#undef RTCX_FOR_EACH_DRIVER_FUNC
#undef RTCX_SYMBOL_NAME
#undef RTCX_STRINGIFY

driver_functions const& driver()
{
  static driver_functions const functions{};
  return functions;
}

// Translates a driver-API status into the closest runtime-API status.
cudaError_t from_driver(CUresult status)
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

// Driver calls resolve against the current context, which the runtime creates lazily;
// `cudaFree(nullptr)` forces it into existence. It runs first so that a missing driver or device is
// reported by the runtime rather than as `cudaErrorInsufficientDriver`.
cudaError_t prepare()
{
  if (auto status = cudaFree(nullptr); status != cudaSuccess) { return status; }
  return driver().loaded ? cudaSuccess : cudaErrorInsufficientDriver;
}

}  // namespace

#endif  // !RTCX_HAS_RUNTIME_LIBRARY_API

cudaError_t library_load_data(cuda_library_t* library, void const* image)
{
#if RTCX_HAS_RUNTIME_LIBRARY_API
  return cudaLibraryLoadData(library, image, nullptr, nullptr, 0, nullptr, nullptr, 0);
#else
  if (auto status = prepare(); status != cudaSuccess) { return status; }
  return from_driver(
    driver().LibraryLoadData(library, image, nullptr, nullptr, 0, nullptr, nullptr, 0));
#endif
}

cudaError_t library_get_kernel(cudaKernel_t* kernel, cuda_library_t library, char const* name)
{
#if RTCX_HAS_RUNTIME_LIBRARY_API
  return cudaLibraryGetKernel(kernel, library, name);
#else
  if (auto status = prepare(); status != cudaSuccess) { return status; }
  return from_driver(driver().LibraryGetKernel(reinterpret_cast<CUkernel*>(kernel), library, name));
#endif
}

cudaError_t library_unload(cuda_library_t library)
{
#if RTCX_HAS_RUNTIME_LIBRARY_API
  return cudaLibraryUnload(library);
#else
  if (!driver().loaded) { return cudaErrorInsufficientDriver; }
  return from_driver(driver().LibraryUnload(library));
#endif
}

cudaError_t launch_kernel(cudaKernel_t kernel,
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
  if (auto status = prepare(); status != cudaSuccess) { return status; }

  // `cuLaunchKernelEx` takes a CUfunction, so resolve the kernel against the current context.
  CUfunction function{};
  if (auto result = driver().KernelGetFunction(&function, reinterpret_cast<CUkernel>(kernel));
      result != CUDA_SUCCESS) {
    return from_driver(result);
  }

  CUlaunchAttribute attribute[1];
  CUlaunchConfig config{};
  config.gridDimX       = grid.x;
  config.gridDimY       = grid.y;
  config.gridDimZ       = grid.z;
  config.blockDimX      = block.x;
  config.blockDimY      = block.y;
  config.blockDimZ      = block.z;
  config.sharedMemBytes = static_cast<unsigned int>(shared_mem);
  config.hStream        = stream;
  config.numAttrs       = 0;
  config.attrs          = nullptr;
  if (cooperative) {
    attribute[0].id                = CU_LAUNCH_ATTRIBUTE_COOPERATIVE;
    attribute[0].value.cooperative = 1;
    config.numAttrs                = 1;
    config.attrs                   = attribute;
  }
  return from_driver(driver().LaunchKernelEx(&config, function, args, nullptr));
#endif
}

}  // namespace rtcx
