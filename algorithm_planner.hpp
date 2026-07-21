/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "algorithm_launcher.hpp"
#include "fragment_entry.hpp"

#include <memory>
#include <shared_mutex>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rtcx {

struct launcher_jit_cache {
  std::shared_mutex mutex;
  std::unordered_map<std::string, std::shared_ptr<algorithm_launcher>> launchers;
};

struct algorithm_planner {
  algorithm_planner(std::string entrypoint, launcher_jit_cache& jit_cache)
    : entrypoint(std::move(entrypoint)), jit_cache_(jit_cache)
  {
  }

  std::shared_ptr<algorithm_launcher> get_launcher();

  std::string entrypoint;
  std::vector<std::unique_ptr<fragment_entry>> fragments;

  template <typename T, typename = std::enable_if_t<std::is_convertible_v<T*, fragment_entry*>>>
  void add_fragment(std::unique_ptr<T> fragment)
  {
    fragments.push_back(std::unique_ptr<fragment_entry>(std::move(fragment)));
  }

  template <typename FragmentTag>
  void add_static_fragment()
  {
    add_fragment(std::make_unique<static_fatbin_fragment_entry<FragmentTag>>());
  }

 protected:
  /** Extra link-time option strings passed to nvJitLink. Base build()
   *  always passes "-lto" and "-arch=sm_XX" first; derived planners may append here in their
   *  constructor body. */
  std::vector<std::string> linktime_extra_options;

 private:
  std::string get_fragments_key() const;
  std::shared_ptr<algorithm_launcher> build();

  std::shared_ptr<algorithm_launcher> read_cache(std::string const& launch_key) const;

  launcher_jit_cache& jit_cache_;
};

}  // namespace rtcx
