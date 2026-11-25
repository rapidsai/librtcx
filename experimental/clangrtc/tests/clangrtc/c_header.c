/**
 * @file c_header.c
 * @brief Compile the public clangrtc header as C and exercise a zero-initialized string view.
 *
 * Exports a small C probe called by the C++ GoogleTest suite to catch C-incompatible public declarations.
 */

#include "clangrtc.h"

/** @brief Verify that a zero-initialized public string view is usable from C. */
int check_c_header(void)
{
    clangrtcString empty = {0};
    return (int)empty.size;
}
