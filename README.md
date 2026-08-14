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
- CUDA >= 12.2

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

  return cache
      .get_or_add_library(make_cache_key(kernel_lto_ir, udf_lto_ir, options),
                          rtcx::library_compile_func::from_functor(link))
      .get();
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
      reinterpret_cast<char const *>(sample_embed::files.data() + range[0]));
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

auto &range = sample_embed::file_ranges[sample_embed::sample_kernel];
auto library =
    rtcx::load_library({sample_embed::files.data() + range[0], range[1]});
auto kernel = library->get_kernel("sample_kernel");
```

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE).
