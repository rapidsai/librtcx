/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* SPDX-License-Identifier: Apache-2.0 */

/**
 * @file artifact_fixture.cpp
 * @brief Prepare self-contained artifacts and remove their original dependency files.
 *
 * Builds source, preprocessed-source, PCH, BMI, and reduced-BMI inputs with retained functions, variables, and kernels.
 */

#include "artifact_fixture.hpp"

#include <format>
#include <fstream>

namespace test
{
bool ArtifactTest::isModule() const
{
    return GetParam() == "bmi" || GetParam() == "reduced-bmi";
}

bool ArtifactTest::isSource() const
{
    return GetParam() == "cuda" || GetParam() == "preprocessed-cuda";
}

std::string ArtifactTest::import() const
{
    return isModule() ? "import sample;\n" : "";
}

void ArtifactTest::SetUp()
{
    CompilationTest::SetUp();
    options.insert(options.end(), {"-O3", "--emit=nvvm-ir-text"});
    auto header = temporary.path / "scale.hpp";
    {
        std::ofstream output(header);
        output << "#define SCALE 3\n";
    }

    prefix = isModule() ? "module;\n#include <clangrtc/device.hpp>\n" : "";
    prefix.append(std::format("#include \"{}\"\n", header.string()));
    prefix.append(R"CUDA(
        namespace example
        {
        template <class T> __device__ T scale(T x)
        {
            return x * SCALE;
        }
        template <class T> __device__ T value = T{7};

        template <class T> __global__ void prepared_kernel(T *out)
        {
            *out = scale(T{9});
        }
        } // namespace example
    )CUDA");

    if (isModule())
    {
        prefix.append(R"CUDA(
            export module sample;

            export namespace example
            {
            using ::example::prepared_kernel;
            using ::example::scale;
            using ::example::value;
            } // namespace example
        )CUDA");
    }

    // This callback is deliberately private to the module. Its prepared IR
    // must survive independently of its visibility to importing consumers.
    prefix.append(R"CUDA(
        namespace internal
        {
        template <class T> __device__ T hidden(T x)
        {
            return x + 1;
        }
        } // namespace internal
    )CUDA");

    if (isSource())
    {
        prefix.append(R"CUDA(
            extern "C" __global__ void kernel(int *out)
            {
                *out = example::scale(5);
            }
        )CUDA");
    }

    preparedExpressions = {"&example::scale<int>", "&example::scale<long long>", "&example::value<int>",
                           "&example::prepared_kernel<int>", "&internal::hidden<int>"};
    auto slots = to_views(preparedExpressions);
    auto input = request(prefix, slots);
    prefixPath = (temporary.path / "prefix.cuh").string();
    input.source.path = to_view(prefixPath);
    auto flags = options;
    flags.push_back(std::format("--artifact={}", GetParam()));
    auto result = prepare(input, flags);
    ASSERT_EQ(result.status, CLANGRTC_SUCCESS) << result.log();
    auto info = result.info();
    ASSERT_EQ(info.symbolCount, preparedExpressions.size());

    for (size_t slot = 0; slot < info.symbolCount; ++slot)
    {
        ASSERT_EQ(to_string(info.symbols[slot].expression), preparedExpressions[slot]);
        ASSERT_NE(info.symbols[slot].name.size, 0);
        preparedNames.push_back(to_string(info.symbols[slot].name));
    }

    package = result.output();
    auto loaded = load(package);
    ASSERT_EQ(loaded.status, CLANGRTC_SUCCESS);
    artifact = std::move(loaded.artifact);
    // Tests consume only packaged dependencies, after original files and the
    // preparation result disappear.
    std::filesystem::remove_all(temporary.path);
}
} // namespace test
