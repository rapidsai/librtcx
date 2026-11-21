# SPDX-License-Identifier: Apache-2.0
#
# @file CUDAHeaderModes.cmake
# @brief Embed feature-gated copies of Clang's CUDA bootstrap for opt-in device header modes.
#
# Keep the installed resource headers untouched. The compiler overlays these copies only for
# --minimal or --stdlib=libcudacxx. Fail configuration if an upstream bootstrap change needs review.

# @brief Replace a reviewed bootstrap fragment and reject unsupported upstream layouts.
function(_clangrtc_gate_cuda_header original replacement)
  string(FIND "${_contents}" "${original}" _position)
  if(_position EQUAL -1)
    message(FATAL_ERROR "CUDA header mode anchor missing in ${_name}: ${original}")
  endif()
  string(REPLACE "${original}" "${replacement}" _contents "${_contents}")
  set(_contents "${_contents}" PARENT_SCOPE)
endfunction()

set(_cuda_modes "\n/** @brief Return feature-gated Clang CUDA resource headers for explicit device modes. */\ninline std::map<std::string,std::string> cuda_mode_headers(){return {\n")
foreach(_name IN ITEMS __clang_cuda_runtime_wrapper.h __clang_cuda_math_forward_declares.h __clang_cuda_intrinsics.h)
  set(_header "${CLANGRTC_RESOURCE_DIR}/include/${_name}")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_header}")
  file(READ "${_header}" _contents)
  if(_name STREQUAL "__clang_cuda_intrinsics.h")
    # This intrinsic returns a plain 32-bit integer; its signature does not need cuda.h.
    _clangrtc_gate_cuda_header("inline cuuint32_t __nvvm_get_smem_pointer"
                              "inline __UINT32_TYPE__ __nvvm_get_smem_pointer")
  elseif(_name STREQUAL "__clang_cuda_runtime_wrapper.h")
    _clangrtc_gate_cuda_header(
      "// Include some forward declares that must come before cmath."
      "#include <stddef.h>\n#include <stdint.h>\n\n// Include some forward declares that must come before cmath.")
    _clangrtc_gate_cuda_header(
      "// Include some forward declares that must come before cmath."
      [=[// CCCL must first see the real CUDA attributes and version, before any C++ headers.
#if defined(__CLANGRTC_LIBCUDACXX__)
#define __CUDACC__ 1
#pragma push_macro("__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__")
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#include "crt/host_defines.h"
#pragma pop_macro("__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__")
#endif

// Include some forward declares that must come before cmath.]=])
    _clangrtc_gate_cuda_header(
      "#include <climits>\n#include <cmath>\n#include <cstdlib>"
      [=[#ifdef __CLANGRTC_LIBCUDACXX__
#include <limits.h>
#include <math.h>
#else
#include <climits>
#include <cmath>
#include <cstdlib>
#endif]=])
    _clangrtc_gate_cuda_header(
      "#include <string.h>"
      [=[#include <string.h>
#ifdef __CLANGRTC_LIBCUDACXX__
// C math macros cannot expand inside cuda::std declarations.
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
#endif]=])
    _clangrtc_gate_cuda_header("#include \"cuda.h\""
                              "#ifndef __CUDACC_RTC_MINIMAL__\n#include \"cuda.h\"\n#endif")
    _clangrtc_gate_cuda_header(
      "#include \"cuda_runtime.h\""
      [=[#ifdef __CUDACC_RTC_MINIMAL__
#include "builtin_types.h"
#include "vector_functions.h"
#else
#include "cuda_runtime.h"
#endif]=])
    _clangrtc_gate_cuda_header(
      "#include \"crt/host_runtime.h\""
      [=[#ifdef __CUDACC_RTC_MINIMAL__
// host_runtime.h normally establishes this prerequisite for crt/math_functions.hpp.
#define __CUDA_INTERNAL_COMPILATION__
#else
#include "crt/host_runtime.h"
#endif]=])
    _clangrtc_gate_cuda_header(
      "// __clang_cuda_texture_intrinsics.h must be included first"
      "#ifndef __CUDACC_RTC_MINIMAL__\n// __clang_cuda_texture_intrinsics.h must be included first")
    _clangrtc_gate_cuda_header("// Restore state of __CUDA_ARCH__"
                              "#endif // !__CUDACC_RTC_MINIMAL__\n\n// Restore state of __CUDA_ARCH__")
    _clangrtc_gate_cuda_header("// We also need device-side std::malloc and std::free."
                              "#ifndef __CLANGRTC_LIBCUDACXX__\n// We also need device-side std::malloc and std::free.")
    _clangrtc_gate_cuda_header("// Out-of-line implementations from __clang_cuda_builtin_vars.h."
                              "#endif // !__CLANGRTC_LIBCUDACXX__\n\n// Out-of-line implementations from __clang_cuda_builtin_vars.h.")
    _clangrtc_gate_cuda_header("#include <__clang_cuda_cmath.h>"
                              "#ifndef __CLANGRTC_LIBCUDACXX__\n#include <__clang_cuda_cmath.h>\n#endif")
    _clangrtc_gate_cuda_header("// curand_mtgp32_kernel helpfully"
                              "#ifndef __CUDACC_RTC_MINIMAL__\n// curand_mtgp32_kernel helpfully")
    # curand_mtgp32_kernel.h pushes __USE_FAST_MATH__; omit its matching pop with the include.
    _clangrtc_gate_cuda_header("#pragma pop_macro(\"__USE_FAST_MATH__\")\n#pragma pop_macro(\"__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__\")"
                              "#pragma pop_macro(\"__USE_FAST_MATH__\")\n#endif // !__CUDACC_RTC_MINIMAL__\n#pragma pop_macro(\"__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__\")")
    _clangrtc_gate_cuda_header("#if CUDA_VERSION >= 9020"
                              "#if CUDA_VERSION >= 9020 && !defined(__CUDACC_RTC_MINIMAL__)")
  else()
    _clangrtc_gate_cuda_header("// We need to define these overloads in exactly the namespace"
                              "#ifndef __CLANGRTC_LIBCUDACXX__\n// We need to define these overloads in exactly the namespace")
    _clangrtc_gate_cuda_header("#pragma pop_macro(\"__DEVICE__\")"
                              "#endif // !__CLANGRTC_LIBCUDACXX__\n\n#pragma pop_macro(\"__DEVICE__\")")
  endif()
  string(APPEND _cuda_modes "{\"${_name}\",R\"CRTC(${_contents})CRTC\"},\n")
endforeach()
string(APPEND _cuda_modes "};}\n")
