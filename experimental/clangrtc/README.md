# clangrtc

clangrtc compiles CUDA device code and native host C/C++ through a C API backed
by Clang/LLVM. It emits LLVM/NVVM bitcode or text, NVIDIA LTO IR, PTX, and native
host objects. The optional curtc component compiles, links, caches, and loads
reference-counted CUDA libraries. Callers own kernel launches and synchronization.

The public headers document options, ownership, diagnostics, and return codes:
[clangrtc.h](include/clangrtc.h) and [curtc.h](include/curtc.h).
`clangrtcString` borrows text or binary bytes with an explicit size; no terminating
NUL is required. Calls are synchronous, and no C++ exception crosses either API.

## Build

Requirements are CMake 3.25+, a C++20 compiler with `std::format`, and zstd development
files for the build-time embed generator.
The default build fetches LLVM/Clang 23.1.5 sources and pinned NVIDIA 13.5.37
compiler libraries. A complete CUDA toolkit is unnecessary for the core build;
CUDA source compilation needs compatible headers/libdevice, and GPU tests need
the real CUDA driver and runtime.

```sh
cmake -S experimental/clangrtc -B build/clangrtc -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCLANGRTC_BUILD_CURTC=ON \
  -DCLANGRTC_CUDA_PATH=/usr/local/cuda
cmake --build build/clangrtc
ctest --test-dir build/clangrtc --output-on-failure
```

The source build keeps LLVM archives and its installation under
`<build>/llvm-downloads`; the inner library build is `<build>/clangrtc-build`.
The outer build delegates installation and test discovery to it.
Build parallelism follows CMake and the selected build tool's defaults.
The source build forwards the settings explicitly listed in
[BuildLLVM.cmake](cmake/BuildLLVM.cmake); put additional target-specific setup
in `CMAKE_TOOLCHAIN_FILE`.

Set `CLANGRTC_LLVM_CACHE` to relocate the cache, or `CLANGRTC_LLVM_SOURCE_DIR`
to use an existing source tree matching the pinned LLVM release. Its version
and published checksum are constants in `cmake/BuildLLVM.cmake` and must be
updated together. `ExternalProject` downloads, verifies, and extracts LLVM
at build time, reusing matching cached archives offline.

To use an existing compiler installation, set `CLANGRTC_FETCH_LLVM=OFF` and
provide `LLVM_DIR` and `Clang_DIR`. Their matching LLVM/Clang 23+ libraries must
be static and position independent. LLVM/Clang are always linked statically;
symbol isolation is not currently performed.

## Dependencies and linkage

libNVVM is shared-only. Our pinned xxHash uses its inline implementation.
The remaining linkage choices are:

| Setting | Effect when enabled |
|---|---|
| `BUILD_SHARED_LIBS` | Build shared clangrtc/curtc libraries; defaults to `ON`. |
| `CLANGRTC_STATIC_LINK_NVJITLINK` | Require nvJitLink and matching nvPTXCompiler archives. |
| `CLANGRTC_STATIC_LINK_CUDART` | Link cudart statically in GPU tests. |

The pinned libNVVM package is shared-only. A static request for any other
dependency fails if the archive is unavailable; it never falls back to a shared
library. C/C++ runtime and LLVM system-dependency linkage remain separate
choices. For example, `ZLIB_USE_STATIC_LIBS=ON` selects static zlib when supported
by the chosen LLVM installation. The CUDA driver is loaded dynamically.

The build always fetches the exact compiler packages listed in
[cuda-compiler-libraries.json](cmake/cuda-compiler-libraries.json) for Linux
x86_64 or AArch64, independently of the application's CUDA runtime. Compiler
libraries are selected for the target CPU architecture; toolkit installations
and backend-path overrides do not select libNVVM or nvJitLink. CUDA source compilation still needs compatible CUDA headers
and libdevice; GPU tests use the toolkit's CUDA runtime.

Downloads live in `CLANGRTC_CUDA_COMPILER_CACHE` and can be reused offline.
Fetched driver headers are private to curtc; their accompanying cudart libraries
are not linked or installed. Installation deploys the fetched compiler libraries
and licenses as needed.

The build fetches Cyan xxHash 0.8.3 with a fixed checksum and embeds its
`XXH_INLINE_ALL` implementation in the libraries. SIMD is selected at compile
time. There is no runtime dispatcher, system xxHash dependency, discovery, or
compilation probe.

Clang's LLVM bitcode dialect must be accepted by the selected libNVVM.
Select compiler libraries for the target platform; configuration does not execute
them on the build machine. Backend tests validate LTO/PTX compilation on the
target platform, and incompatible custom combinations report compilation errors.
Changing IR text or version metadata cannot make a modern module compatible
with an older reader.

## Compile and prepare

The CLI lives in the inner build directory when LLVM is built from source:

```sh
build/clangrtc/clangrtc-build/clangrtc compile kernel.cu -o kernel.lto -- \
  --cuda-path=/usr/local/cuda --gpu-architecture=compute_86 --lto-ir
```

Specify only one output option; repeated selections are errors:

| Option | Output |
|---|---|
| `--emit=nvvm-ir-bitcode` | NVVM-compatible LLVM bitcode; device default. |
| `--emit=nvvm-ir-text` | Textual NVVM-compatible LLVM IR. |
| `--lto-ir` | NVIDIA LTO bitcode through libNVVM. |
| `--ptx-text` | PTX through libNVVM. |
| `--emit=llvm-ir-bitcode` / `--emit=llvm-ir-text` | Native host LLVM IR in host mode. |
| `--emit=object` | Position-independent native host object. |

`--host-only` selects host C++20; `--source=c` and `--source=c++` select host
C17 and C++20. Override the standard with `-std=...`. Host compilation uses
system headers and needs no CUDA path. Linking and executing host objects remain
caller-owned; mixed host/device translation units and cross-target code generation
are outside this API.

```sh
clangrtc compile host.cpp -o host.o -- --host-only --emit=object
clangrtc prepare prefix.cppm -o prefix.artifact -- --artifact=reduced-bmi
clangrtc compile consumer.cu --artifact-file prefix.artifact -o kernel.bc
clangrtc inspect prefix.artifact
```

Preparation supports `cuda`, `preprocessed-cuda`, `pch`, `bmi`, and `reduced-bmi`.
Host C++ supports PCH and both BMI formats; host C supports PCH. Packages own
their dependencies and optional prepared definitions. Load an artifact once
with `clangrtcLoadArtifact`, keep it alive during borrowing requests, and release
it with `clangrtcDestroyArtifact`. Each compilation creates a fresh frontend;
immutable artifacts can be shared across concurrent requests.

Consumers inherit preparation options when none are supplied. Explicit options
must match the semantic environment; output selection and NVVM tuning may differ.
PCH/BMI compatibility requires a matching compiler build and configuration.
Only load serialized Clang state from trusted producers.

Expression slots select functions, kernels, and variables to retain and report
their resolved names. Preparation and compilation have independent slot lists.
C expressions use their existing C ABI names unchanged.

## Device headers and options

Device compilation uses CUDA toolkit headers and Clang's compiler identity.
The `nvrtc` and `hosted` profile names select the same environment; Clang does
not define `__CUDACC_RTC__`. BMI producers include `<clangrtc/device.hpp>` after
`module;` so the CUDA bootstrap follows the global-module-fragment declaration.

`--minimal` omits automatic texture/surface and CUDA runtime API declarations.
`--stdlib=libcudacxx` forwards supported standard headers to freestanding
`cuda::std` without host C++ headers. Pass a CCCL include root supporting
`_CCCL_ENABLE_FREESTANDING` with `-I`. These modes are independent and retained
in prepared artifacts; real CUDA math, atomics, and intrinsics remain available.

CUDA requests use `-nocudainc` and explicitly load the embedded
[device.hpp](headers/clangrtc/device.hpp) bootstrap. The default environment reuses
Clang's original CUDA wrapper. Device header modes establish their own include
order using unmodified Clang and CUDA implementation headers. Minimal mode also
overlays `cuda_runtime_api.h` with an empty file to prevent transitive runtime API
imports. The device header modes require CUDA 9 or newer.

`rtcx_embed` links the workaround headers and matching Clang resource headers into
clangrtc with `COMPRESSION none`. The compiler reads them through a virtual
filesystem: users need neither their original files nor a resource-directory option.
The explicit lists in [EmbedHeaders.cmake](cmake/EmbedHeaders.cmake) contain
CUDA/NVPTX support, CUDA C++ wrappers, and shared portable C definitions.
Update those lists when adding workaround headers or upgrading LLVM.
`CLANGRTC_RESOURCE_DIR` selects the resource headers at build time;
`--resource-dir=PATH` can override the bundle when additional compiler resources,
such as CPU-specific intrinsic headers, are needed.
CUDA toolkit and host standard-library headers remain external. zstd is required
by the build-time generator; the header bundle is uncompressed at runtime.

Repeat `--clang-option=ARG` or `--nvvm-option=ARG` to forward multiple options.
The two-entry forms are also accepted; arguments are never shell-split.
Target, output, backend selection, and compiler-pipeline controls are reserved.
NVVM options apply to LTO/PTX output and do not invalidate prepared frontend
state. `-nvvm-latest` is selected by clangrtc.

`clangrtcGetVersion` accepts `clang`, `llvm`, and `nvvm`; the CLI equivalent is
`clangrtc --version`. NVVM reports its API version, not its CUDA package version.
`clangrtcInit` can initialize the compiler backend explicitly; otherwise it is
initialized lazily when needed. Initialization is thread-safe and retryable.

## curtc libraries

Enable `CLANGRTC_BUILD_CURTC` and create a `curtcCache` with its configuration.
Pass ordered source, prepared-artifact, LTO, PTX, or cubin inputs to `curtcGet`.
One request can select multiple kernels or link fragments without CUDA source.
Per-call statistics distinguish cache hits, compilation, linking, and loading.

The cache handles memory/disk reuse and concurrent work sharing. Returned
`curtcLibrary` references own loaded CUDA libraries independently of cache
lifetime and eviction. Enumerate selected metadata with `curtcEnumerateKernels`
and retrieve exported `CUkernel` handles by symbol with `curtcGetKernel`.
Retain a library until all GPU work using its kernels has completed.

curtc requires CUDA 12.4+ driver APIs and adds nvJitLink; it does not link cudart.
`curtcInit` optionally initializes its backends and driver without creating a
CUDA context. Callers own contexts, streams, launches, and synchronization.

## Install, test, and document

```sh
cmake --install build/clangrtc --prefix /path/to/install
```

Consumers use `find_package(clangrtc CONFIG REQUIRED)` and link
`clangrtc::clangrtc` or `clangrtc::curtc`. Static consumers also need the matching
LLVM/Clang installation. Required NVIDIA archives are installed with the
package; nvPTXCompiler is fetched only for static nvJitLink.

`BUILD_TESTING=ON` enables GoogleTest; select its package with `GTest_DIR` or
disable tests to build without it. Compiler and cache suites run without a GPU.
`CURTC_BUILD_GPU_TESTS=ON` adds real driver/runtime execution tests. Use
`CUDA_VISIBLE_DEVICES` to select the GPU; tests use visible ordinal zero.

`CLANGRTC_BUILD_DOCS=ON` enables public-header documentation. After the default
source build, run `cmake --build build/clangrtc/clangrtc-build --target clangrtc_docs`.
With a prebuilt LLVM installation, run that target in the library build directory.
Doxygen 1.9+ generates HTML and XML under `<inner-build>/docs`; documentation
warnings are errors. C/C++ and CMake formatting follow the local configuration files.
