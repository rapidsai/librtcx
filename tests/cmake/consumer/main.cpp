/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "consumer_embed.hpp"

#include <rtcx/rtcx.hpp>

#include <cstddef>

int main() { return std::size(consumer_embed::file_ranges) > 0 ? 0 : 1; }
