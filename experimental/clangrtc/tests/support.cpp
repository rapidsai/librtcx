/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* SPDX-License-Identifier: Apache-2.0 */

/**
 * @file support.cpp
 * @brief Implement shared compiler-call helpers and temporary test resources.
 *
 * Copies diagnostics and result views while keeping C API handles alive and removes temporary files after each fixture.
 */

#include "support.hpp"

#include "test_config.hpp"

#include <atomic>
#include <chrono>
#include <format>
#include <stdexcept>
#include <system_error>

namespace test
{
clangrtcString to_view(std::string_view value)
{
    return {value.data(), value.size()};
}

std::string to_string(clangrtcString value)
{
    return value.size ? std::string(value.data, value.size) : std::string{};
}

std::vector<clangrtcString> to_views(std::span<std::string const> strings)
{
    std::vector<clangrtcString> result;

    for (auto const &string : strings)
    {
        result.push_back(to_view(string));
    }

    return result;
}

clangrtcRequest request(std::string_view source, std::span<clangrtcString const> expressions)
{
    clangrtcRequest result{};
    result.source = {to_view("/__clangrtc__/test.cu"), to_view(source)};
    result.expressions = expressions.data();
    result.expressionCount = expressions.size();
    return result;
}

clangrtcResultView Operation::info() const
{
    clangrtcResultView value{};

    if (clangrtcGetResult(result.get(), &value) != CLANGRTC_SUCCESS)
    {
        throw std::runtime_error("No result view available");
    }

    return value;
}

std::string Operation::log() const
{
    return to_string(info().log);
}

std::string Operation::output() const
{
    return to_string(info().output);
}

Operation compile(clangrtcRequest const &input, Options const &options, clangrtcDiagnosticCallback callback,
                  void *userData)
{
    auto flags = to_views(options);
    clangrtcResult raw = nullptr;
    auto status = clangrtcCompile(&input, flags.data(), flags.size(), callback, userData, &raw);
    return {status, Result(raw, clangrtcDestroyResult)};
}

Operation compile(clangrtcRequest const &input, Options const &options)
{
    return compile(input, options, nullptr, nullptr);
}

Operation prepare(clangrtcRequest const &input, Options const &options)
{
    auto flags = to_views(options);
    clangrtcResult raw = nullptr;
    auto status = clangrtcPrepare(&input, flags.data(), flags.size(), nullptr, nullptr, &raw);
    return {status, Result(raw, clangrtcDestroyResult)};
}

LoadedArtifact load(std::string_view package)
{
    clangrtcArtifact raw = nullptr;
    auto status = clangrtcLoadArtifact(to_view(package), nullptr, nullptr, &raw);
    return {status, Artifact(raw, clangrtcDestroyArtifact)};
}

TemporaryDirectory::TemporaryDirectory()
    : path([] {
          static std::atomic<unsigned> sequence{};
          return std::filesystem::temp_directory_path() /
                 std::format("clangrtc-test-{}-{}", std::chrono::steady_clock::now().time_since_epoch().count(),
                             sequence.fetch_add(1));
      }())
{
    std::filesystem::create_directories(path);
}

TemporaryDirectory::~TemporaryDirectory()
{
    std::error_code ignored;
    std::filesystem::remove_all(path, ignored);
}

void CompilationTest::SetUp()
{
    options = {std::format("--cuda-path={}", CudaPath), "--gpu-architecture=compute_86"};
}
} // namespace test
