/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#define SET_ERROR_MSG(msg, location_prefix, fmt, ...)                                            \
  do {                                                                                           \
    int size1 = std::snprintf(nullptr, 0, "%s", location_prefix);                                \
    int size2 = std::snprintf(nullptr, 0, "file=%s line=%d: ", __FILE__, __LINE__);              \
    int size3 = std::snprintf(nullptr, 0, fmt, ##__VA_ARGS__);                                   \
    if (size1 < 0 || size2 < 0 || size3 < 0)                                                     \
      throw std::runtime_error("Error in snprintf, cannot format rtcx exception.");              \
    auto size = size1 + size2 + size3 + 1; /* +1 for final '\0' */                               \
    std::vector<char> buf(size);                                                                 \
    std::snprintf(buf.data(), size1 + 1 /* +1 for '\0' */, "%s", location_prefix);               \
    std::snprintf(                                                                               \
      buf.data() + size1, size2 + 1 /* +1 for '\0' */, "file=%s line=%d: ", __FILE__, __LINE__); \
    std::snprintf(buf.data() + size1 + size2, size3 + 1 /* +1 for '\0' */, fmt, ##__VA_ARGS__);  \
    msg += std::string(buf.data(), buf.data() + size - 1); /* -1 to remove final '\0' */         \
  } while (0)

#define RTCX_CUDA_TRY(call)                        \
  do {                                             \
    cudaError_t const status = call;               \
    if (status != cudaSuccess) {                   \
      cudaGetLastError();                          \
      std::string msg{};                           \
      SET_ERROR_MSG(msg,                           \
                    "CUDA error encountered at: ", \
                    "call='%s', Reason=%s:%s",     \
                    #call,                         \
                    cudaGetErrorName(status),      \
                    cudaGetErrorString(status));   \
      throw std::runtime_error(msg);               \
    }                                              \
  } while (0)

#define RTCX_EXPECTS(cond, fmt, ...)                              \
  do {                                                            \
    if (!(cond)) {                                                \
      std::string msg{};                                          \
      SET_ERROR_MSG(msg, "RTCX failure at ", fmt, ##__VA_ARGS__); \
      throw std::logic_error(msg);                                \
    }                                                             \
  } while (0)

#define RTCX_FAIL(fmt, ...)                                     \
  do {                                                          \
    std::string msg{};                                          \
    SET_ERROR_MSG(msg, "RTCX failure at ", fmt, ##__VA_ARGS__); \
    throw std::logic_error(msg);                                \
  } while (0)
