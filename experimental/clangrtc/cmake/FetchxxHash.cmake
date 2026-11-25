# SPDX-License-Identifier: Apache-2.0
#
# @file FetchxxHash.cmake
#
# @brief Fetch our pinned xxHash header for private inline hashing.

include(FetchContent)
FetchContent_Declare(
    clangrtc_xxhash
    URL "https://codeload.github.com/Cyan4973/xxHash/tar.gz/refs/tags/v0.8.3"
    URL_HASH "SHA256=aae608dfe8213dfd05d909a57718ef82f30722c392344583d3f39050c7f29a80"
    DOWNLOAD_DIR "${CMAKE_CURRENT_BINARY_DIR}/_deps"
    DOWNLOAD_NAME xxHash-v0.8.3.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP FALSE
    TLS_VERIFY ON)
FetchContent_MakeAvailable(clangrtc_xxhash)

install(FILES "${clangrtc_xxhash_SOURCE_DIR}/LICENSE"
        DESTINATION "${CMAKE_INSTALL_DATADIR}/clangrtc/licenses/xxHash")
