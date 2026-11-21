/* SPDX-License-Identifier: Apache-2.0 */

/**
 * @file curtc_c_header.c
 * @brief Create a loaded library through the curtc C API from a C translation unit.
 *
 * Supplies a C fixture to the GoogleTest suite, which asserts metadata, kernel lookup,
 * and library lifetime after the C-owned cache has been destroyed.
 */

#include <curtc.h>

/**
 * @brief Create a real CUDA library through C and release its cache before returning.
 * @param image Borrowed valid cubin bytes produced for the actual GPU.
 * @param output Destination cleared on entry; receives one library reference on success.
 * @return Status of the first failed C API operation, or CLANGRTC_SUCCESS.
 */
clangrtcStatus create_curtc_library_from_c(clangrtcString image, curtcLibrary *output)
{
    *output = NULL;
    curtcConfiguration configuration = {0};
    curtcCache cache = NULL;
    curtcRequest request = {0};
    curtcInput input = {0};
    request.inputs = &input;
    request.inputCount = 1;
    clangrtcStatus status = curtcDefaultConfiguration(&configuration);

    if (status != CLANGRTC_SUCCESS)
    {
        return status;
    }

    configuration.diskCache = false;
    configuration.memoryCache = false;
    status = curtcCreateCache(&configuration, &cache);

    if (status != CLANGRTC_SUCCESS)
    {
        return status;
    }

    input.type = CURTC_INPUT_TYPE_CUBIN;
    input.data = image;
    curtcKernelSelection selected = {{"c_kernel", 8}, {"app-kernel", 10}};
    input.entries = &selected;
    input.entryCount = 1;
    status = curtcGet(cache, &request, NULL, NULL, output, NULL);
    curtcDestroyCache(cache);
    return status;
}
