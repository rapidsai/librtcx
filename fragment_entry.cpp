/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <fragment_entry.hpp>

namespace rtcx {

bool fatbin_fragment_entry::add_to(nvJitLinkHandle& handle) const
{
  auto result = nvJitLinkAddData(handle, NVJITLINK_INPUT_ANY, get_data(), get_length(), get_key());

  check_nvjitlink_result(handle, result);
  return true;
}

}  // namespace rtcx
