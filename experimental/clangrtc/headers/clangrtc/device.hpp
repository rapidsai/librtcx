/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file device.hpp
 * @brief Import Clang's CUDA environment inside a module's global fragment.
 *
 * Include after `module;` when preparing a CUDA BMI. Ordinary CUDA compilations
 * receive the same wrapper automatically from Clang's driver.
 */
#pragma once

#include <__clang_cuda_runtime_wrapper.h>
