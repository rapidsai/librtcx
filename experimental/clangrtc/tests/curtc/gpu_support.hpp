/* SPDX-License-Identifier: Apache-2.0 */

/**
 * @file gpu_support.hpp
 * @brief Real CUDA resources shared by the curtc integration tests.
 */
#pragma once

#include <cuda.h>
#include <string>
#include <vector>

namespace test
{
/** @brief Return the selected real GPU's sm_NN compilation target. */
std::string const &gpu_target();

/** @brief Compile a real cubin exporting first, second, fixed, and c_kernel, each writing value. */
std::vector<char> cubin_fixture(int value);

/** @brief Compile the standard cubin fixture whose kernels write 42. */
std::vector<char> cubin_fixture();

/** @brief Launch a kernel taking int* through the real driver and check its result using cudart. */
void launch(CUkernel kernel, int expected);

/** @brief Launch a kernel taking float* through the real driver and check its result using cudart. */
void launch(CUkernel kernel, float expected);
} // namespace test
