/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rtcx/embed_codec.hpp>

#define XXH_INLINE_ALL
#include <xxhash.h>
#include <zstd.h>

#include <format>
#include <stdexcept>

namespace rtcx::embed::detail {

std::vector<std::uint8_t> zstd_compress(std::span<std::uint8_t const> bytes)
{
  std::vector<std::uint8_t> compressed(ZSTD_compressBound(bytes.size()));
  auto const compressed_size =
    ZSTD_compress(compressed.data(), compressed.size(), bytes.data(), bytes.size(), 22);
  if (ZSTD_isError(compressed_size)) {
    throw std::runtime_error(
      std::format("Compression failed with error: {}", ZSTD_getErrorName(compressed_size)));
  }
  compressed.resize(compressed_size);
  return compressed;
}

rtcx::hash128 xxh3_128(std::span<std::span<std::uint8_t const> const> parts)
{
  XXH3_state_t state;
  XXH3_INITSTATE(&state);
  XXH3_128bits_reset(&state);
  for (auto part : parts) {
    XXH3_128bits_update(&state, part.data(), part.size());
  }
  auto hash = XXH3_128bits_digest(&state);
  return rtcx::hash128{hash.high64, hash.low64};
}

}  // namespace rtcx::embed::detail
