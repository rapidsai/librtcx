/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace rtcx {

/**
 * @brief A 128-bit hash represented as a hexadecimal string.
 */
struct [[nodiscard]] hash128_hex_string {
  static constexpr std::size_t NUM_HEX_DIGITS =
    32;  ///< The number of hexadecimal digits in a 128-bit hash string
  static constexpr std::size_t NUM_HEX_BYTES =
    NUM_HEX_DIGITS / 2;  ///< The number of bytes in a 128-bit hash

  char data_[NUM_HEX_DIGITS + 1];  // NOLINT(modernize-avoid-c-arrays)

  /**
   * @brief Returns a string view of the hash128_hex_string.
   * @return A string view representing the hash128_hex_string.
   */
  [[nodiscard]] constexpr std::string_view view() const
  { return std::string_view{data_, NUM_HEX_DIGITS}; }

  /**
   * @brief Implicit conversion operator to std::string_view.
   * @return A string view representing the hash128_hex_string.
   */
  [[nodiscard]] constexpr operator std::string_view() const { return view(); }

  /**
   * @brief Returns a pointer to the underlying character array of the hash128_hex_string.
   * @return A pointer to the character array representing the hash128_hex_string.
   */
  [[nodiscard]] char const* data() const;

  /**
   * @brief Returns a null-terminated C-style string representation of the hash128_hex_string.
   * @return A pointer to the null-terminated character array representing the hash128_hex_string
   */
  [[nodiscard]] char const* c_str() const;

  /**
   * @brief Returns the size of the hash128_hex_string in characters (excluding the null
   * terminator).
   * @return The size of the hash128_hex_string in characters.
   */
  [[nodiscard]] static constexpr std::size_t size() { return NUM_HEX_DIGITS; }

  /**
   * @brief Creates a hash128_hex_string from a span of bytes.
   * @param input A span of bytes representing the hash.
   * @return A hash128_hex_string representing the hash.
   */
  static hash128_hex_string make(std::span<std::uint8_t const, NUM_HEX_BYTES> input);

  /**
   * @brief Creates a hash128_hex_string from a 128-bit unsigned integer.
   * @param hash A 128-bit unsigned integer representing the hash.
   * @return A hash128_hex_string representing the hash.
   */
  static hash128_hex_string make(__uint128_t hash);
};

/**
 * @brief A 128-bit hash represented as a 128-bit unsigned integer.
 * @param value The 128-bit unsigned integer representing the hash.
 * @return A hash128 object representing the hash.
 */
struct hash128 {
  __uint128_t value;  ///< The 128-bit unsigned integer representing the hash

  /**
   * @brief Constructs a hash128 object.
   * @param v The 128-bit unsigned integer representing the hash.
   */
  constexpr hash128(__uint128_t v = 0) : value(v) {}

  /**
   * @brief Constructs a hash128 object from high and low 64-bit parts.
   * @param high The high 64 bits of the 128-bit hash.
   * @param low The low 64 bits of the 128-bit hash.
   */
  constexpr hash128(std::uint64_t high, std::uint64_t low)
    : value((static_cast<__uint128_t>(high) << 64) | low)
  {
  }

  /**
   * @brief Compares two hash128 objects for equality.
   * @param other The other hash128 object to compare with.
   * @return true if the two hash128 objects are equal, false otherwise.
   */
  [[nodiscard]] constexpr bool operator==(hash128 const& other) const = default;

  /**
   * @brief Accesses the byte at the specified index in the hash128 object.
   * @param index The index of the byte to access (0-based).
   * @return The byte at the specified index.
   */
  [[nodiscard]] std::uint8_t operator[](std::size_t index) const;

  /**
   * @brief Returns the size of the hash128 object in bytes.
   * @return The size of the hash128 object in bytes.
   */
  [[nodiscard]] std::size_t size() const;

  /**
   * @brief Returns a pointer to the underlying byte array of the hash128 object.
   * @return A pointer to the byte array representing the hash128 object.
   */
  [[nodiscard]] std::uint8_t const* data() const;

  /**
   * @brief Converts the hash128 object to a hash128_hex_string representation.
   * @return A hash128_hex_string representing the hash128 object.
   */
  hash128_hex_string to_hex_string() const;

  /**
   * @brief Parses a hash128_hex_string and returns a hash128 object.
   * @param hex A string view representing the hash128_hex_string to parse.
   * @return A hash128 object representing the parsed hash.
   */
  static hash128 parse(std::string_view hex);
};

}  // namespace rtcx
