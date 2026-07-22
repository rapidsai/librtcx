/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <nvJitLink.h>
#include <rtcx/nvjitlink_checker.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

namespace rtcx {

struct fragment_entry {
  virtual ~fragment_entry() = default;

  virtual bool add_to(nvJitLinkHandle& handle) const = 0;

  virtual char const* get_key() const = 0;
};

struct fatbin_fragment_entry : fragment_entry {
  virtual uint8_t const* get_data() const = 0;

  virtual size_t get_length() const = 0;

  bool add_to(nvJitLinkHandle& handle) const override final;
};

template <typename FragmentTag>
struct static_fatbin_fragment_entry final : fatbin_fragment_entry {
  uint8_t const* get_data() const override
  {
    return static_fatbin_fragment_entry<FragmentTag>::data;
  }

  size_t get_length() const override { return static_fatbin_fragment_entry<FragmentTag>::length; }

  char const* get_key() const override
  {
    return typeid(static_fatbin_fragment_entry<FragmentTag>).name();
  }

  static uint8_t const* const data;
  static size_t const length;
};

struct udf_fatbin_fragment final : fatbin_fragment_entry {
  udf_fatbin_fragment(std::string key, std::vector<uint8_t> bytes)
    : key_(std::move(key)), bytes_(std::move(bytes))
  {
  }

  uint8_t const* get_data() const override { return bytes_.data(); }

  size_t get_length() const override { return bytes_.size(); }

  char const* get_key() const override { return key_.c_str(); }

 private:
  std::string key_;
  std::vector<uint8_t> bytes_;
};

}  // namespace rtcx
