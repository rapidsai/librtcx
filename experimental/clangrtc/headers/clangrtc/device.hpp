/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */

/**
 * @file device.hpp
 * @brief Establish the CUDA device environment without Clang's automatic pre-include.
 *
 * Ordinary requests pre-include this header explicitly. BMI producers include it
 * after `module;` so declarations stay inside the global module fragment.
 * The device header modes reuse Clang and CUDA implementation headers with an
 * explicit initialization order. Bootstrap adaptations derive from LLVM 23.1.5.
 */
#pragma once

#if !defined(__CUDACC_RTC_MINIMAL__) && !defined(__CLANGRTC_LIBCUDACXX__)
#include <__clang_cuda_runtime_wrapper.h>
#else
#define __CLANG_CUDA_RUNTIME_WRAPPER_H__

// Fundamental types must precede CUDA and C++ library declarations.
#include <stddef.h>
#include <stdint.h>

#ifdef __CUDACC_RTC_MINIMAL__
/** @brief Driver integer type required by Clang intrinsics when cuda.h is omitted. */
using cuuint32_t = __UINT32_TYPE__;
#endif

#if defined(__CLANGRTC_LIBCUDACXX__)
#define __CUDACC__ 1
#pragma push_macro("__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__")
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#include "crt/host_defines.h"
#pragma pop_macro("__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__")
#endif

#ifndef __CLANGRTC_LIBCUDACXX__
#include <__clang_cuda_math_forward_declares.h>
#endif

#define __CUDACC__ 1

#ifdef __CLANGRTC_LIBCUDACXX__
#include <limits.h>
#include <math.h>
#else
#include <climits>
#include <cmath>
#include <cstdlib>
#endif
#include <stdlib.h>
#include <string.h>
#ifdef __CLANGRTC_LIBCUDACXX__
// C macros must not expand inside cuda::std math declarations.
#undef fpclassify
#undef isfinite
#undef isinf
#undef isnan
#undef isnormal
#undef signbit
#undef isgreater
#undef isgreaterequal
#undef isless
#undef islessequal
#undef islessgreater
#undef isunordered
#endif
#undef __CUDACC__

#if defined(__GNUC__) && defined(__GLIBC_PREREQ)
#if __GLIBC_PREREQ(2, 42)
#define _NV_RSQRT_SPECIFIER noexcept(true)
#endif
#endif

#ifndef _NV_RSQRT_SPECIFIER
#define _NV_RSQRT_SPECIFIER
#endif

// SDK headers temporarily change these definitions while importing device code.
#pragma push_macro("__THROW")
#pragma push_macro("__CUDA_ARCH__")

#ifndef __CUDACC_RTC_MINIMAL__
#include "cuda.h"
#endif
#if !defined(CUDA_VERSION)
#error "cuda.h did not define CUDA_VERSION"
#elif CUDA_VERSION < 9000
#error "clangrtc device header modes require CUDA 9 or newer"
#endif

#pragma push_macro("__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__")
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__

#include "__clang_cuda_builtin_vars.h"

// Clang owns builtin variables and libdevice wrappers; suppress SDK redeclarations.
#define __DEVICE_LAUNCH_PARAMETERS_H__

#define __DEVICE_FUNCTIONS_H__
#define __MATH_FUNCTIONS_H__
#define __COMMON_FUNCTIONS_H__
#define __DEVICE_FUNCTIONS_DECLS_H__
// Clang supplies vote intrinsics; the SDK implementation below supplies atomics.
#define __DEVICE_ATOMIC_FUNCTIONS_H__

#undef __CUDACC__
#define __CUDACC__ 1
#define __CUDA_LIBDEVICE__
#include "host_defines.h"
#undef __CUDACC__
#include "driver_types.h"
#include "host_config.h"

#pragma push_macro("nv_weak")
#define nv_weak weak
#undef __CUDABE__
#undef __CUDA_LIBDEVICE__
#define __CUDACC__ 1
#ifdef __CUDACC_RTC_MINIMAL__
#include "builtin_types.h"
#include "vector_functions.h"
#else
#include "cuda_runtime_api.h"
#include "vector_functions.h"
#endif

#pragma pop_macro("nv_weak")
#undef __CUDACC__
#define __CUDABE__

#define __nvvm_memcpy(s, d, n, a) __builtin_memcpy(s, d, n)
#define __nvvm_memset(d, c, n, a) __builtin_memset(d, c, n)

#ifdef __CUDACC_RTC_MINIMAL__
// math_functions.hpp expects the prerequisite normally supplied by host_runtime.h.
#define __CUDA_INTERNAL_COMPILATION__
#else
#include "crt/host_runtime.h"
#endif
#undef __cxa_vec_ctor
#undef __cxa_vec_cctor
#undef __cxa_vec_dtor
#undef __cxa_vec_new
#undef __cxa_vec_new2
#undef __cxa_vec_new3
#undef __cxa_vec_delete2
#undef __cxa_vec_delete
#undef __cxa_vec_delete3
#undef __cxa_pure_virtual

// Reuse the compiler's complete implementations rather than maintaining a catalog.
#include <__clang_cuda_libdevice_declares.h>

#include <__clang_cuda_device_functions.h>
#include <__clang_cuda_math.h>

#undef __THROW
#define __THROW

#if defined(CU_DEVICE_INVALID)
#if !defined(__USE_FAST_MATH__)
#define __USE_FAST_MATH__ 0
#endif

#if !defined(__CUDA_PREC_DIV)
#define __CUDA_PREC_DIV 0
#endif
#endif

#pragma push_macro("__host__")
#define __host__ UNEXPECTED_HOST_ATTRIBUTE

#pragma push_macro("__forceinline__")
#define __forceinline__ __device__ __inline__ __attribute__((always_inline))

#pragma push_macro("__USE_FAST_MATH__")
#if defined(__CLANG_GPU_APPROX_TRANSCENDENTALS__)
#define __USE_FAST_MATH__ 1
#endif

#include "crt/math_functions.hpp"

#pragma pop_macro("__USE_FAST_MATH__")

#pragma pop_macro("__forceinline__")

#undef __MATH_FUNCTIONS_HPP__
#undef __host__
#define __host__
#undef __CUDABE__
#define __CUDACC__ 1
#undef __DEVICE_FUNCTIONS_HPP__
#include "device_atomic_functions.hpp"
#include "crt/device_functions.hpp"
#include "crt/device_double_functions.hpp"
#include "sm_20_atomic_functions.hpp"
// SDK address-space helpers use volatile assembly; retain Clang's pure builtins.
#pragma push_macro("__isGlobal")
#pragma push_macro("__isShared")
#pragma push_macro("__isConstant")
#pragma push_macro("__isLocal")
#define __isGlobal __ignored_cuda___isGlobal
#define __isShared __ignored_cuda___isShared
#define __isConstant __ignored_cuda___isConstant
#define __isLocal __ignored_cuda___isLocal
#include "sm_20_intrinsics.hpp"
#pragma pop_macro("__isGlobal")
#pragma pop_macro("__isShared")
#pragma pop_macro("__isConstant")
#pragma pop_macro("__isLocal")
#pragma push_macro("__DEVICE__")
#define __DEVICE__ static __device__ __forceinline__ __attribute__((const))

/** @brief Test whether a pointer belongs to global device memory. */
__DEVICE__ unsigned int __isGlobal(const void *p)
{
    return __nvvm_isspacep_global(p);
}

/** @brief Test whether a pointer belongs to shared device memory. */
__DEVICE__ unsigned int __isShared(const void *p)
{
    return __nvvm_isspacep_shared(p);
}

/** @brief Test whether a pointer belongs to constant device memory. */
__DEVICE__ unsigned int __isConstant(const void *p)
{
    return __nvvm_isspacep_const(p);
}

/** @brief Test whether a pointer belongs to local device memory. */
__DEVICE__ unsigned int __isLocal(const void *p)
{
    return __nvvm_isspacep_local(p);
}

#pragma pop_macro("__DEVICE__")
#include "sm_32_atomic_functions.hpp"

#pragma push_macro("__CUDA_ARCH__")
#undef __CUDA_ARCH__
#include "sm_60_atomic_functions.hpp"
#include "sm_61_intrinsics.hpp"
#pragma pop_macro("__CUDA_ARCH__")

#undef __MATH_FUNCTIONS_HPP__

// Reload device overloads without colliding with host standard-library signbit.
#pragma push_macro("signbit")
#pragma push_macro("__GNUC__")
#undef __GNUC__
#define signbit __ignored_cuda_signbit

#pragma push_macro("_GLIBCXX_MATH_H")
#pragma push_macro("_LIBCPP_VERSION")
#undef _GLIBCXX_MATH_H
#ifdef _LIBCPP_VERSION
#define _LIBCPP_VERSION 3700
#endif

#include "crt/math_functions.hpp"
#pragma pop_macro("_GLIBCXX_MATH_H")
#pragma pop_macro("_LIBCPP_VERSION")
#pragma pop_macro("__GNUC__")
#pragma pop_macro("signbit")

#pragma pop_macro("__host__")

#ifndef __CUDACC_RTC_MINIMAL__
#if __cplusplus >= 201103L && CUDA_VERSION >= 9000
#include <__clang_cuda_texture_intrinsics.h>
#else
/** @brief Delay the texture-language diagnostic until an unsupported call is instantiated. */
template <typename T> struct __nv_tex_needs_cxx11
{
    /** @brief Always false for the unsupported language-mode diagnostic. */
    const static bool value = false;
};

/** @brief Reject texture operations when the selected C++ dialect cannot support them. */
template <class T>
__host__ __device__ void __nv_tex_surf_handler(const char *name, T *ptr, cudaTextureObject_t obj, float x)
{
    _Static_assert(__nv_tex_needs_cxx11<T>::value, "Texture support requires C++11");
}
#endif // __cplusplus >= 201103L && CUDA_VERSION >= 9000
#include "surface_indirect_functions.h"
#if CUDA_VERSION < 13000
#include "texture_fetch_functions.h"
#endif // CUDA_VERSION < 13000
#include "texture_indirect_functions.h"

#endif // !__CUDACC_RTC_MINIMAL__

#pragma pop_macro("__CUDA_ARCH__")
#pragma pop_macro("__THROW")

#undef __CUDABE__
#define __CUDACC__ 1

// Device ABI entry points are supplied by the driver; assert forwards to __assertfail.
extern "C"
{
    /** @brief Write formatted device output through the CUDA device ABI. */
    __device__ int vprintf(const char *, const char *);

    /** @brief Release device heap storage through the CUDA device ABI. */
    __device__ void free(void *) __attribute((nothrow));

    /** @brief Allocate device heap storage through the CUDA device ABI. */
    __device__ void *malloc(size_t) __attribute((nothrow)) __attribute__((malloc));

    /** @brief Report a failed device assertion through the CUDA device ABI. */
    __device__ void __assertfail(const char *__message, const char *__file, unsigned __line, const char *__function,
                                 size_t __charSize);

    /** @brief Forward a C library assertion to the CUDA device ABI. */
    __device__ static inline void __assert_fail(const char *__message, const char *__file, unsigned __line,
                                                const char *__function)
    {
        __assertfail(__message, __file, __line, __function, sizeof(char));
    }

    /** @brief Declare device printf for Clang to lower to vprintf. */
    __device__ int printf(const char *, ...);
} // extern "C"

#ifdef __CLANGRTC_LIBCUDACXX__
/** @brief Classify NaNs without importing host C++ math declarations. */
template <class T> __host__ __device__ inline bool isnan(T value) noexcept
{
    if constexpr (__is_integral(T))
    {
        return false;
    }
    else
    {
        return __builtin_isnan(value);
    }
}

/** @brief Classify infinities without importing host C++ math declarations. */
template <class T> __host__ __device__ inline bool isinf(T value) noexcept
{
    if constexpr (__is_integral(T))
    {
        return false;
    }
    else
    {
        return __builtin_isinf(value);
    }
}

/** @brief Classify finite values without importing host C++ math declarations. */
template <class T> __host__ __device__ inline bool isfinite(T value) noexcept
{
    if constexpr (__is_integral(T))
    {
        return true;
    }
    else
    {
        return __builtin_isfinite(value);
    }
}
#endif

#ifndef __CLANGRTC_LIBCUDACXX__
namespace std
{
/** @brief Forward std::free to device heap deallocation. */
__device__ static inline void free(void *__ptr)
{
    ::free(__ptr);
}

/** @brief Forward std::malloc to device heap allocation. */
__device__ static inline void *malloc(size_t __size)
{
    return ::malloc(__size);
}
} // namespace std

#endif // !__CLANGRTC_LIBCUDACXX__

/** @brief Convert the builtin threadIdx value to CUDA dim3. */
__device__ inline __cuda_builtin_threadIdx_t::operator dim3() const
{
    return dim3(x, y, z);
}

/** @brief Convert the builtin threadIdx value to CUDA uint3. */
__device__ inline __cuda_builtin_threadIdx_t::operator uint3() const
{
    return {x, y, z};
}

/** @brief Convert the builtin blockIdx value to CUDA dim3. */
__device__ inline __cuda_builtin_blockIdx_t::operator dim3() const
{
    return dim3(x, y, z);
}

/** @brief Convert the builtin blockIdx value to CUDA uint3. */
__device__ inline __cuda_builtin_blockIdx_t::operator uint3() const
{
    return {x, y, z};
}

/** @brief Convert the builtin blockDim value to CUDA dim3. */
__device__ inline __cuda_builtin_blockDim_t::operator dim3() const
{
    return dim3(x, y, z);
}

/** @brief Convert the builtin blockDim value to CUDA uint3. */
__device__ inline __cuda_builtin_blockDim_t::operator uint3() const
{
    return {x, y, z};
}

/** @brief Convert the builtin gridDim value to CUDA dim3. */
__device__ inline __cuda_builtin_gridDim_t::operator dim3() const
{
    return dim3(x, y, z);
}

/** @brief Convert the builtin gridDim value to CUDA uint3. */
__device__ inline __cuda_builtin_gridDim_t::operator uint3() const
{
    return {x, y, z};
}

#ifndef __CLANGRTC_LIBCUDACXX__
#include <__clang_cuda_cmath.h>
#endif
#include <__clang_cuda_intrinsics.h>
#include <__clang_cuda_complex_builtins.h>

#pragma pop_macro("__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__")

#if CUDA_VERSION >= 9020 && !defined(__CUDACC_RTC_MINIMAL__)
/** @brief Declare CUDA's launch-configuration entry point for nonminimal device compilation. */
extern "C" unsigned __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem = 0, void *stream = 0);
#endif

#define _CG_CLUSTER_INTRINSICS_AVAILABLE 1

#ifndef __CUDACC_RTC_MINIMAL__
// C++ adapters may include CCCL, which needs the address-space intrinsics above.
#include "cuda_runtime.h"
#endif

#endif
