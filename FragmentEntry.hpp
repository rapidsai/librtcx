/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "nvjitlink_checker.hpp"

#include <nvJitLink.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <typeinfo>
#include <vector>

struct FragmentEntry {
  virtual ~FragmentEntry() = default;

  virtual bool add_to(nvJitLinkHandle& handle) const = 0;

  virtual char const* get_key() const = 0;
};

struct FatbinFragmentEntry : FragmentEntry {
  virtual uint8_t const* get_data() const = 0;

  virtual size_t get_length() const = 0;

  bool add_to(nvJitLinkHandle& handle) const override final;
};

template <typename FragmentTag>
struct StaticFatbinFragmentEntry final : FatbinFragmentEntry {
  uint8_t const* get_data() const override { return StaticFatbinFragmentEntry<FragmentTag>::data; }

  size_t get_length() const override { return StaticFatbinFragmentEntry<FragmentTag>::length; }

  char const* get_key() const override
  {
    return typeid(StaticFatbinFragmentEntry<FragmentTag>).name();
  }

  static uint8_t const* const data;
  static size_t const length;
};

struct UDFFatbinFragment final : FatbinFragmentEntry {
  UDFFatbinFragment(std::string key, std::vector<uint8_t> bytes)
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
