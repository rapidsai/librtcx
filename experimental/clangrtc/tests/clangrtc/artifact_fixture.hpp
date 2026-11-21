/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* SPDX-License-Identifier: Apache-2.0 */

/**
 * @file artifact_fixture.hpp
 * @brief Declare the fixture shared by all serialized-artifact format tests.
 *
 * Owns prepared packages, retained expression names, and a dependency tree removed before artifact reuse.
 */

#pragma once
#include "support.hpp"

namespace test
{
class ArtifactTest : public CompilationTest, public ::testing::WithParamInterface<std::string>
{
  protected:
    std::string prefix;

    std::string prefixPath;

    std::string package;

    std::vector<std::string> preparedNames;

    std::vector<std::string> preparedExpressions;

    Artifact artifact{nullptr, clangrtcDestroyArtifact};

    void SetUp() override;

    bool isModule() const;

    bool isSource() const;

    std::string import() const;
};

} // namespace test
