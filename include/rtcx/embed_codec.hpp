/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <rtcx/hash.hpp>

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace rtcx::embed::detail {

/**
 * @brief Compress bytes into a single zstd frame at the configured compression level.
 *
 * @param bytes Bytes to compress
 * @return The compressed frame
 * @throws std::runtime_error if compression fails
 */
std::vector<std::uint8_t> zstd_compress(std::span<std::uint8_t const> bytes);

/**
 * @brief Compute the XXH3 128-bit hash of the concatenation of several byte ranges.
 *
 * @param parts Byte ranges hashed in order as one stream
 * @return The 128-bit hash
 */
rtcx::hash128 xxh3_128(std::span<std::span<std::uint8_t const> const> parts);

}  // namespace rtcx::embed::detail
