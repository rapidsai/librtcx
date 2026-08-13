/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <rtcx/fragment_entry.hpp>

#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rtcx {

struct nvrtc_lto_fragment_compiler {
  nvrtc_lto_fragment_compiler();

  std::vector<std::string> standard_compile_opts;
  std::unordered_map<std::string, std::vector<uint8_t>> cache;
  mutable std::shared_mutex cache_mutex_;

  std::unique_ptr<udf_fatbin_fragment> compile(std::string const& key, std::string const& code);

 private:
  std::unique_ptr<udf_fatbin_fragment> read_cache(std::string const& key) const;
};

nvrtc_lto_fragment_compiler& nvrtc_compiler();

}  // namespace rtcx
