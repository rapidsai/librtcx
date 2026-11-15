# librtcx

RTCX (runtime-compiler extended) is a C++20 wrapper around NVRTC and nvJitLink. It provides:

- User-controlled compilation, linking, caching, and preloading of CUDA kernels
- Zero-copy interfaces for JIT compilation and linking
- CMake utilities for embedding optionally compressed headers and binary artifacts
- Explicit loading and unloading of `libcuda`, `libnvrtc`, and `libnvJitLink`
- Composition-oriented APIs that leave policy decisions to the application

## Platforms Supported
- Linux

## Build-time Requirements
- CMake >= 4.0
- libzstd
- xxHash
- CUDA >= 12.9

# Dependencies
- nvJitlink >= 12.2
- NVRTC >= 12.2
- LibCUDA >= 12.2

## Building

```bash
cmake -S . -B build
cmake --build build --parallel
cmake --install build --prefix /path/to/install
```

Build the C++ API documentation with Doxygen:

```bash
cmake --build build --target rtcx_docs
```

The generated HTML is written to `build/docs/html`. Run `./ci/checks/doxygen.sh` to verify that
Doxygen completes without documentation warnings.

By default, librtcx links dynamically to NVRTC and nvJitLink. Set `RTCX_STATIC_LINK_NVRTC=ON` and/or `RTCX_STATIC_LINK_NVJITLINK=ON` to use the static toolkit libraries instead.

Consume an installed package with:

```cmake
find_package(rtcx CONFIG REQUIRED)
target_link_libraries(my_target PRIVATE rtcx::rtcx)
```

## Examples

The cache examples use an application-provided `make_cache_key(...)` helper. The key must include every input that can affect the output: source or binary content, options, target architecture, and relevant toolchain versions. A collision-resistant content hash such as XXH3-128 is suitable.

### 1. CUDA C++ JIT Compilation and Caching

This compiles CUDA source to a CUBIN and caches the CUBIN on disk and loaded library in memory:

```cpp
rtcx::hash128 make_cache_key(std::string_view source,
                             std::span<char const *const> options);

int main() {
  rtcx::initialize();

  constexpr char source[] = R"cuda(
    extern "C" __global__ void sample_kernel(void *) {}
  )cuda";
  char const *options[] = {"--gpu-architecture=sm_80"};

  // Both directories must exist, be writable, and be on the same filesystem.
  std::filesystem::create_directories("/tmp/rtcx-cache/objects");
  std::filesystem::create_directories("/tmp/rtcx-cache/staging");
  rtcx::cache_t cache{"/tmp/rtcx-cache/objects", "/tmp/rtcx-cache/staging",
                      rtcx::cache_limits{},
                      /* preload = */ false,
                      /* disable = */ false};

  auto compile = [&]() -> std::tuple<rtcx::library, rtcx::blob> {
    auto binary =
        std::make_shared<rtcx::blob_t>(rtcx::blob_t::from_buffer(rtcx::compile({
            .name = "sample.cu",
            .source = source,
            .options = options,
            .target_type = rtcx::binary_type::CUBIN,
        })));
    return {rtcx::load_library(binary->view()), binary};
  };

  auto library =
      cache
          .get_or_add_library(make_cache_key(source, options),
                              rtcx::library_compile_func::from_functor(compile))
          .get();
  auto kernel = library->get_kernel("sample_kernel");

  void *kernel_params[] = {nullptr};
  kernel.launch(/* grid_dim = */ {1, 1, 1}, /* block_dim = */ {1, 1, 1}, 0,
                /* stream = */ nullptr, kernel_params);
}
```

### 2. CUDA JIT Linking and Caching

Precompiled fragments can be linked with nvJitLink and cached in the same way:

```cpp
extern std::span<std::uint8_t const> kernel_lto_ir;
extern std::span<std::uint8_t const> udf_lto_ir;

rtcx::hash128 make_cache_key(std::span<std::uint8_t const> kernel,
                             std::span<std::uint8_t const> udf,
                             std::span<char const *const> options);

int main() {
  rtcx::initialize();

  rtcx::memory_fragment fragments[] = {
      {.data = kernel_lto_ir,
       .type = rtcx::binary_type::LTO_IR,
       .name = "kernel.ltoir"},
      {.data = udf_lto_ir,
       .type = rtcx::binary_type::LTO_IR,
       .name = "udf.ltoir"},
  };
  char const *options[] = {"-lto", "-arch=sm_80"};

  // Both directories must exist, be writable, and be on the same filesystem.
  std::filesystem::create_directories("/tmp/rtcx-cache/objects");
  std::filesystem::create_directories("/tmp/rtcx-cache/staging");
  rtcx::cache_t cache{"/tmp/rtcx-cache/objects", "/tmp/rtcx-cache/staging",
                      rtcx::cache_limits{},
                      /* preload = */ false,
                      /* disable = */ false};

  auto link = [&]() -> std::tuple<rtcx::library, rtcx::blob> {
    auto binary = std::make_shared<rtcx::blob_t>(
        rtcx::blob_t::from_buffer(rtcx::link_library({
            .name = "sample-linked-library",
            .output_type = rtcx::binary_type::CUBIN,
            .memory_fragments = fragments,
            .link_options = options,
        })));
    return {rtcx::load_library(binary->view()), binary};
  };

  auto library =
      cache
          .get_or_add_library(
              make_cache_key(kernel_lto_ir, udf_lto_ir, options),
              rtcx::library_compile_func::from_functor(link))
          .get();

  return 0;
}
```

`rtcx::link_params` also accepts file-backed inputs through `file_fragments` and can produce PTX instead of a CUBIN.

### 3. Source-code Embedding

The embedding utilities can help package CUDA headers into an executable:

```cmake
cmake_minimum_required(VERSION 4.0)
project(sample LANGUAGES CXX ASM)

find_package(rtcx CONFIG REQUIRED)

rtcx_add_embed(sample_embed)
rtcx_embed_includes(
  sample_embed
  SOURCE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include"
  DEST_DIRECTORY include
  INCLUDE_DIRECTORIES include
)
rtcx_embed(
  sample_embed
  COMPRESSION none
  OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated"
)

add_executable(sample main.cpp "${sample_embed_SOURCE_DIR}/sample_embed.s")
add_dependencies(sample sample_embed)
target_include_directories(sample PRIVATE "${sample_embed_INCLUDE_DIRS}")
target_link_libraries(sample PRIVATE rtcx::rtcx)
```

The generated header describes the files and virtual paths.
Pass pointers into the embedded data to NVRTC:

```cpp
#include "sample_embed.hpp"
#include <rtcx/rtcx.hpp>

#include <cstddef>
#include <vector>

std::vector<char const *> headers;
headers.reserve(std::size(sample_embed::file_ranges));
for (auto &range : sample_embed::file_ranges) {
  headers.push_back(
      reinterpret_cast<char const *>(range(sample_embed::files).data()));
}

char const *options[] = {"--gpu-architecture=sm_80", "--include-path=include"};
auto binary = rtcx::compile({
    .name = "jit-source.cu",
    .source = R"cuda(#include <my_header.cuh>)cuda",
    .header_include_names = sample_embed::file_destinations,
    .headers = headers,
    .options = options,
    .target_type = rtcx::binary_type::CUBIN,
});
```

Omit `FILES` from `rtcx_embed_includes` to embed the source directory recursively, or provide `FILES` to select paths.

### 4. CUDA Binary Embedding

`rtcx_embed_blob` embeds an artifact such as a CUBIN, FATBIN, PTX file, or LTO IR file:

```cmake
rtcx_add_embed(sample_embed)
rtcx_embed_blob(
  sample_embed
  ID sample_kernel
  TAG "sample/kernel"
  FILE "${CMAKE_CURRENT_SOURCE_DIR}/sample_kernel.fatbin"
  DEST sample_kernel.fatbin
)
rtcx_embed(
  sample_embed
  COMPRESSION none
  OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated"
)

add_executable(sample main.cpp "${sample_embed_SOURCE_DIR}/sample_embed.s")
add_dependencies(sample sample_embed)
target_include_directories(sample PRIVATE "${sample_embed_INCLUDE_DIRS}")
target_link_libraries(sample PRIVATE rtcx::rtcx)
```

Each blob ID becomes an index in the generated namespace:

```cpp
#include "sample_embed.hpp"
#include <rtcx/rtcx.hpp>

auto library = rtcx::load_library(sample_embed::get(sample_embed::sample_kernel));
auto kernel = library->get_kernel("sample_kernel");

// Compile-time lookup by integer ID or the optional TAG string:
auto by_id = sample_embed::get<sample_embed::sample_kernel>();
auto by_tag = sample_embed::get<"sample/kernel">();
```

For `COMPRESSION none`, `get(index)` returns a const reference to a cached `std::span<std::uint8_t const>` over
the selected file's bytes, excluding its appended null terminator. Pass a valid
file ID from the generated namespace. For compressed embeds, `get(index)` returns
a const reference to the `range_t` containing `{offset, size}` in `file_ranges`. These offsets refer to
the decompressed bytes; decompress `files` before applying the range. Call
`range(bytes)` on a span to obtain its subspan.

The template overloads return the same const reference as `get(index)`, and reject
out-of-range IDs and unknown tags at compile time. `TAG` is optional and separate
from `ID`; each nonempty tag must be unique within its embed target. Tags may
contain characters such as `/` that are not valid in C++ identifiers. Files
without tags remain accessible by integer ID.
