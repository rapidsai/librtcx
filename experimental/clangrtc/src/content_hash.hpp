/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file content_hash.hpp
 * @brief Shared XXH3-128 content identities using our private inline xxHash implementation.
 */
#pragma once

#include <format>
#include <span>
#include <string>

#define XXH_INLINE_ALL
#include <xxhash.h>

namespace cr
{
/** @brief Canonical hexadecimal width of an XXH3-128 digest. */
inline constexpr size_t content_hash_hex_size = sizeof(XXH128_hash_t) * 2;

/**
 * @brief Compute an XXH3-128 digest as canonical lowercase hexadecimal text.
 * @param bytes Exact borrowed content, including embedded zero bytes.
 * @return High word followed by low word, independent of host endianness.
 */
inline std::string content_hash(std::span<char const> bytes)
{
    auto value = XXH3_128bits(bytes.data(), bytes.size());
    constexpr auto word_width = sizeof(value.high64) * 2;
    return std::format("{:0{}x}{:0{}x}", value.high64, word_width, value.low64, word_width);
}
} // namespace cr
