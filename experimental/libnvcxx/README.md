# nvcxx

A lightweight C++ 20 standard library for CUDA kernels.


## Requirements

- NVRTC >=12.9

# Setup

- NVRTC Flags:
    - `-std=c++20`
    - `--device-int128`: enables the required 128-bit integer support.
    - `--device-as-default-execution-space`: treats unannotated library functions as device functions.
    - `-DNVCXX_AS_STD` (To make `nvcxx` to populate the `cuda::std`, `cuda::` and `std::` namespaces, optional)
    - `-Ilibnvcxx`
    - `-Ilibnvcxx/nvcxx`
    - `-Ilibnvcxx/nvcxx/cuda`
    - `-Ilibnvcxx/nvcxx/cuda/std`
    - `--minimal`


## Why libnvcxx

Full libcudacxx serves a broad compiler, platform, and C++ standard-library
compatibility surface. Our target is a small set of device operations compiled
by one compiler family. The extra generality raises the cost of compiling each
new kernel:

- **Heavy transitive headers.** A small operation can bring in configuration,
  traits, concepts, utility, and other implementation headers. Unused template
  bodies can avoid instantiation while their active declarations still need
  preprocessing and parsing.

- **Compatibility machinery outside our target.** Compiler and host-library
  detection, portability macros, language-version gates, workarounds, and
  fallback implementations support environments this project does not target.
  The larger source and include graph still need preprocessing; disabled
  branches are not all semantically analyzed or instantiated.

- **More complete standard-library contracts.** Allocator-aware tuple
  construction, conditional triviality and exception specifications, optional
  lifetime machinery, and extensive constrained overload sets add declarations,
  trait queries, and instantiation work beyond the operations our kernels use.

- **Costs remain with PCH.** A precompiled header reduces repeated parsing, but
  restoring compiler state, instantiating the kernel's types/functions, and
  optimizing device code still take time.
