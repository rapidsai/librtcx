/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file cli.cpp
 * @brief Command-line preparation, compilation, and artifact inspection.
 */

#include "clangrtc.h"

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

/** @brief Borrow a string's bytes for a synchronous C API call. */
static clangrtcString to_view(std::string const &s)
{
    return {s.data(), s.size()};
}

/**
 * @brief Execute the requested compiler or inspection command.
 * @param argc Number of command-line arguments.
 * @param argv Command name and arguments.
 * @return Zero on success, one on failure, or two for invalid usage.
 */
int main(int argc, char **argv)
{
    if (argc == 2 && std::string_view(argv[1]) == "--version")
    {
        for (auto name : {"clang", "llvm", "nvvm"})
        {
            clangrtcString version{};
            std::string component(name);
            auto status = clangrtcGetVersion(to_view(component), &version);
            std::cout << name << ": ";

            if (status == CLANGRTC_SUCCESS)
            {
                std::cout.write(version.data, version.size);
            }
            else
            {
                std::cout << "unavailable";
            }

            std::cout << '\n';
        }

        return 0;
    }

    if (argc < 3)
    {
        std::cerr << "usage: clangrtc prepare|compile INPUT -o OUTPUT [--artifact-file "
                     "FILE] -- OPTIONS...\n       clangrtc inspect PACKAGE\n"
                     "device outputs: --lto-ir, --ptx-text, --emit=nvvm-ir-bitcode|nvvm-ir-text\n"
                     "host: --host-only or --source=c|c++, --emit=llvm-ir-bitcode|llvm-ir-text|object\n"
                     "repeatable: --clang-option=ARG, --nvvm-option=ARG\n"
                     "versions: clangrtc --version\n";
        return 2;
    }

    std::string command = argv[1], input = argv[2], output, artifactPath;
    std::vector<std::string> opts;
    bool compilerOptions = false;

    for (int i = 3; i < argc; ++i)
    {
        std::string arg = argv[i];

        if (arg == "--")
        {
            compilerOptions = true;
        }
        else if (!compilerOptions && arg == "-o" && i + 1 < argc)
        {
            output = argv[++i];
        }
        else if (!compilerOptions && arg == "--artifact-file" && i + 1 < argc)
        {
            artifactPath = argv[++i];
        }
        else
        {
            opts.push_back(arg);
        }
    }

    std::vector<clangrtcString> options;

    for (auto const &s : opts)
    {
        options.push_back(to_view(s));
    }

    clangrtcArtifact a = nullptr;
    clangrtcResult resultHandle = nullptr;
    // Views returned by the C API borrow these handles, so release them only
    // after diagnostics, output bytes, and metadata have been consumed.
    auto finish = [&](int code) {
        clangrtcDestroyResult(resultHandle);
        clangrtcDestroyArtifact(a);
        return code;
    };

    auto read = [](std::string const &path) {
        std::ifstream f(path, std::ios::binary);

        if (!f)
        {
            throw std::runtime_error("Cannot read " + path);
        }

        return std::string(std::istreambuf_iterator<char>(f), {});
    };

    try
    {
        auto source = read(input);

        if (command == "inspect")
        {
            auto status = clangrtcLoadArtifact(to_view(source), nullptr, nullptr, &a);

            if (status)
            {
                std::cerr << "Invalid or incompatible artifact\n";
                return finish(1);
            }

            clangrtcArtifactInfo info{};
            clangrtcGetArtifactInfo(a, &info);
            std::cout << "format: " << std::string(info.format.data, info.format.size)
                      << "\ntarget: " << std::string(info.target.data, info.target.size)
                      << "\npackage bytes: " << info.packageSize << "\ndependencies: " << info.dependencyCount << '\n';

            for (size_t i = 0; i < info.dependencyCount; ++i)
            {
                std::cout << std::string(info.dependencies[i].data, info.dependencies[i].size) << '\n';
            }

            return finish(0);
        }

        if (command != "prepare" && command != "compile")
        {
            std::cerr << "Unknown command\n";
            return finish(2);
        }

        if (output.empty())
        {
            std::cerr << "Specify -o OUTPUT\n";
            return finish(2);
        }

        if (!artifactPath.empty())
        {
            auto bytes = read(artifactPath);

            if (clangrtcLoadArtifact(to_view(bytes), nullptr, nullptr, &a))
            {
                std::cerr << "Cannot load artifact\n";
                return finish(1);
            }
        }

        clangrtcRequest q{};
        q.source = {to_view(input), to_view(source)};
        q.artifact = a;
        auto status = command == "prepare"
                          ? clangrtcPrepare(&q, options.data(), options.size(), nullptr, nullptr, &resultHandle)
                          : clangrtcCompile(&q, options.data(), options.size(), nullptr, nullptr, &resultHandle);
        clangrtcResultView result{};
        clangrtcGetResult(resultHandle, &result);

        if (result.log.size)
        {
            std::cerr.write(result.log.data, result.log.size);
        }

        if (status)
        {
            return finish(1);
        }

        std::ofstream file(output, std::ios::binary);
        file.write(result.output.data, result.output.size);

        if (!file)
        {
            return finish(1);
        }

        for (size_t i = 0; i < result.timingCount; ++i)
        {
            std::cerr << std::string(result.timings[i].stage.data, result.timings[i].stage.size) << ": "
                      << result.timings[i].elapsedNs / 1.e6 << " ms\n";
        }

        return finish(0);
    }
    catch (std::exception const &e)
    {
        std::cerr << e.what() << '\n';
        return finish(1);
    }
}
