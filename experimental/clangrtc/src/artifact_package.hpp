/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file artifact_package.hpp
 * @brief Read the serialized artifact envelope without copying or hashing its payload.
 */
#pragma once

#include "content_hash.hpp"
#include "error.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <limits>
#include <span>
#include <string_view>

namespace cr
{
/** @brief Shared package and prepared-state version; bump for incompatible format or compiler-convention changes. */
inline constexpr std::string_view artifact_version = "CLANGRTC-27.1-ARTIFACT-1";

/** @brief Number of hexadecimal characters in the serialized XXH3-128 checksum. */
inline constexpr size_t artifact_checksum_size = content_hash_hex_size;

/** @brief Borrowed framing fields; their lifetime is the supplied package's lifetime. */
struct artifact_envelope
{
    /** @brief Stored digest of the manifest and concatenated payloads. */
    std::string_view checksum;

    /** @brief Manifest and payload bytes covered by checksum. */
    std::span<char const> contents;

    /** @brief Length of the JSON manifest at the start of contents. */
    size_t manifest_size;
};

/**
 * @brief Read bounded artifact framing and its stored content identity.
 * @param package Borrowed serialized artifact bytes.
 * @return Nonowning envelope with a syntactically valid XXH3-128 checksum.
 * @throws Failure If the framing, checksum spelling, or manifest length is invalid.
 * @note This does not verify payload integrity. Loading validates the full checksum;
 *       cache lookups may use this identity directly for valid, trusted packages.
 */
inline artifact_envelope read_artifact_envelope(std::span<char const> package)
{
    constexpr auto length_digits = std::numeric_limits<size_t>::digits10 + 1;
    constexpr auto max_header_size =
        artifact_version.size() + sizeof('\n') + length_digits + sizeof('\n') + artifact_checksum_size + sizeof('\n');
    std::string_view header(package.data(), std::min(package.size(), max_header_size));
    size_t consumed{};
    auto field = [&] {
        auto end = header.find('\n');
        CR_EXPECTS(end != std::string_view::npos, "Truncated artifact header", Failure, CLANGRTC_INVALID_ARTIFACT);
        auto value = header.substr(0, end);
        header.remove_prefix(end + 1);
        consumed += end + 1;
        return value;
    };

    CR_EXPECTS(field() == artifact_version, "Not a clangrtc artifact", Failure, CLANGRTC_INVALID_ARTIFACT);
    auto length = field();
    size_t manifest_size{};
    auto parsed = std::from_chars(length.data(), length.data() + length.size(), manifest_size);
    CR_EXPECTS(parsed.ec == std::errc{} && parsed.ptr == length.data() + length.size(), "Invalid manifest length",
               Failure, CLANGRTC_INVALID_ARTIFACT);
    auto checksum = field();
    CR_EXPECTS(
        checksum.size() == artifact_checksum_size &&
            std::ranges::all_of(checksum, [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); }),
        "Invalid artifact checksum", Failure, CLANGRTC_INVALID_ARTIFACT);
    auto contents = package.subspan(consumed);
    CR_EXPECTS(manifest_size <= contents.size(), "Truncated manifest", Failure, CLANGRTC_INVALID_ARTIFACT);
    return {checksum, contents, manifest_size};
}
} // namespace cr
