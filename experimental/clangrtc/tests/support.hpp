/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* SPDX-License-Identifier: Apache-2.0 */

/**
 * @file support.hpp
 * @brief Declare shared fixtures and ownership helpers for compiler and runtime tests.
 *
 * Provides borrowed string adapters, result and artifact owners, temporary directories, and default CUDA options.
 */

#pragma once

#include "clangrtc.h"

#include <filesystem>
#include <gtest/gtest.h>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace test
{
using Result = std::unique_ptr<clangrtcResultImpl, decltype(&clangrtcDestroyResult)>;

using Artifact = std::unique_ptr<clangrtcArtifactImpl, decltype(&clangrtcDestroyArtifact)>;

using Options = std::vector<std::string>;

clangrtcString to_view(std::string_view text);

std::string to_string(clangrtcString value);

std::vector<clangrtcString> to_views(std::span<std::string const> strings);

clangrtcRequest request(std::string_view source, std::span<clangrtcString const> expressions);

struct Operation
{
    clangrtcStatus status;

    Result result;

    clangrtcResultView info() const;

    std::string log() const;

    std::string output() const;
};

Operation compile(clangrtcRequest const &input, Options const &options);

Operation compile(clangrtcRequest const &input, Options const &options, clangrtcDiagnosticCallback callback,
                  void *userData);

Operation prepare(clangrtcRequest const &input, Options const &options);

struct LoadedArtifact
{
    clangrtcStatus status;

    Artifact artifact;
};

LoadedArtifact load(std::string_view package);

class TemporaryDirectory
{
  public:
    std::filesystem::path const path;

    TemporaryDirectory();

    ~TemporaryDirectory();

    TemporaryDirectory(TemporaryDirectory const &) = delete;

    TemporaryDirectory &operator=(TemporaryDirectory const &) = delete;
};

class CompilationTest : public ::testing::Test
{
  protected:
    Options options;

    TemporaryDirectory temporary;

    void SetUp() override;
};

} // namespace test
