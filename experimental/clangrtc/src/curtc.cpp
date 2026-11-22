/* SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file curtc.cpp
 * @brief Cached CUDA compilation, linking, and reference-counted kernel loading.
 *
 * The public C API owns independent caches and loaded CUDA libraries. Internal
 * cache tests reuse the byte/cache implementation with CURTC_INTERNAL_TESTS.
 */
#include "artifact_package.hpp"
#include "error.hpp"
#include "function_ref.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <clangrtc.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cuda.h>
#include <curtc.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <nvJitLink.h>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unistd.h>
#include <utility>
#include <variant>
#include <vector>

/**
 * @brief Apply a macro to every required CUDA driver entry point.
 * @param entry Macro accepting the full external entry-point name.
 */
// clang-format off
#define CR_CUDA_ENTRIES(entry)           \
    entry(cuInit)                        \
    entry(cuLibraryLoadData)             \
    entry(cuLibraryGetKernelCount)       \
    entry(cuLibraryEnumerateKernels)     \
    entry(cuKernelGetName)               \
    entry(cuLibraryUnload)               \
    entry(cuGetErrorString)
// clang-format on

/**
 * @brief Declare a function pointer with its external C API name and ABI.
 * @param name Full external entry-point name.
 */
#define CR_DECLARE_API_ENTRY(name) decltype(&::name) name{};

/**
 * @brief Resolve a declared CUDA driver function pointer from the owned library.
 * @param name Full external entry-point name.
 */
#define CR_LOAD_CUDA_ENTRY(name) name = symbol<decltype(name)>(#name);

/**
 * @brief Apply a macro to every required nvJitLink entry point.
 * @param entry Macro accepting the full external entry-point name.
 */
// clang-format off
#define CR_NVJITLINK_ENTRIES(entry)            \
    entry(nvJitLinkCreate)                     \
    entry(nvJitLinkDestroy)                    \
    entry(nvJitLinkAddData)                    \
    entry(nvJitLinkComplete)                   \
    entry(nvJitLinkGetLinkedCubinSize)         \
    entry(nvJitLinkGetLinkedCubin)             \
    entry(nvJitLinkGetErrorLogSize)            \
    entry(nvJitLinkGetErrorLog)                \
    entry(nvJitLinkVersion)
// clang-format on

/**
 * @brief Bind a declared nvJitLink function pointer according to the configured linkage.
 * @param name Full external entry-point name.
 */
#if CLANGRTC_STATIC_LINK_NVJITLINK
#define CR_LOAD_NVJITLINK_ENTRY(name) name = &::name;
#else
#define CR_LOAD_NVJITLINK_ENTRY(name) name = symbol<decltype(name)>(#name);
#endif

/**
 * @brief Check a CUDA driver call, preserving its expression, call site, and error text.
 * @param operation CUDA expression evaluated exactly once.
 * @param get_error_string Resolved cuGetErrorString pointer, used only on failure.
 * @throws cr::error with toolchain-error status and the cuda diagnostic stage.
 */
#define CR_CHECK_CUDA(operation, get_error_string)                                                                     \
    do                                                                                                                 \
    {                                                                                                                  \
        auto const cr_cuda_status = (operation);                                                                       \
                                                                                                                       \
        if (cr_cuda_status != CUDA_SUCCESS)                                                                            \
        {                                                                                                              \
            char const *cr_cuda_message{};                                                                             \
            (get_error_string)(cr_cuda_status, &cr_cuda_message);                                                      \
            throw ::cr::error("cuda",                                                                                  \
                              std::format("clangrtc failure at {}:{}: CUDA call `{}` failed ({}): {}", __FILE__,       \
                                          __LINE__, #operation, static_cast<int>(cr_cuda_status),                      \
                                          cr_cuda_message ? cr_cuda_message : "Unknown CUDA error"),                   \
                              CLANGRTC_TOOLCHAIN_ERROR);                                                               \
        }                                                                                                              \
    } while (false)

/**
 * @brief Check an nvJitLink call, preserving its expression, call site, and error log.
 * @param operation nvJitLink expression evaluated exactly once.
 * @param link_handle Handle read on failure after evaluating operation; may be null.
 * @throws cr::error with link-failed status and the nvjitlink diagnostic stage.
 */
#define CR_CHECK_NVJITLINK(operation, link_handle)                                                                     \
    do                                                                                                                 \
    {                                                                                                                  \
        auto const cr_nvjitlink_status = (operation);                                                                  \
                                                                                                                       \
        if (cr_nvjitlink_status != NVJITLINK_SUCCESS)                                                                  \
        {                                                                                                              \
            auto const cr_nvjitlink_handle = (link_handle);                                                            \
            size_t cr_nvjitlink_size{};                                                                                \
            std::string cr_nvjitlink_log;                                                                              \
                                                                                                                       \
            if (cr_nvjitlink_handle &&                                                                                 \
                ::cr::nvJitLink().nvJitLinkGetErrorLogSize(cr_nvjitlink_handle, &cr_nvjitlink_size) ==                 \
                    NVJITLINK_SUCCESS &&                                                                               \
                cr_nvjitlink_size)                                                                                     \
            {                                                                                                          \
                cr_nvjitlink_log.resize(cr_nvjitlink_size);                                                            \
                                                                                                                       \
                if (::cr::nvJitLink().nvJitLinkGetErrorLog(cr_nvjitlink_handle, cr_nvjitlink_log.data()) !=            \
                    NVJITLINK_SUCCESS)                                                                                 \
                {                                                                                                      \
                    cr_nvjitlink_log.clear();                                                                          \
                }                                                                                                      \
                                                                                                                       \
                if (!cr_nvjitlink_log.empty() && cr_nvjitlink_log.back() == '\0')                                      \
                {                                                                                                      \
                    cr_nvjitlink_log.pop_back();                                                                       \
                }                                                                                                      \
            }                                                                                                          \
                                                                                                                       \
            throw ::cr::error("nvjitlink",                                                                             \
                              std::format("clangrtc failure at {}:{}: nvJitLink call `{}` failed ({}): {}", __FILE__,  \
                                          __LINE__, #operation, static_cast<int>(cr_nvjitlink_status),                 \
                                          cr_nvjitlink_log),                                                           \
                              CLANGRTC_LINK_FAILED);                                                                   \
        }                                                                                                              \
    } while (false)

namespace cr
{

/**
 * @brief Cache identity schema; bump when key encoding or compilation semantics change.
 */
constexpr std::string_view cache_identity_version = "curtc-7";

/**
 * @brief Versioned disk format marker shared by cache readers and writers.
 */
constexpr std::string_view format_magic = "CURTC-CACHE-2\n";

/**
 * @brief Suffix shared by cache filename creation and pruning.
 */
constexpr std::string_view file_extension = ".bin";

/**
 * @brief Owns shared memory retention and persistent stage-result caching.
 */
class content_cache;

/**
 * @brief Hash an exact byte range as canonical XXH3-128 hexadecimal text.
 * @param s Bytes to hash, including any embedded NULs.
 * @return Content digest used for cache keys and integrity checks.
 */
inline std::string digest(std::span<char const> s)
{
    return content_hash(s);
}

/**
 * @brief Convert bytes to bytes.
 */
constexpr std::uint64_t operator""_B(unsigned long long value)
{
    return value;
}

/**
 * @brief Convert kilobytes to bytes.
 */
constexpr std::uint64_t operator""_KB(unsigned long long value)
{
    return value * 1000_B;
}

/**
 * @brief Convert megabytes to bytes.
 */
constexpr std::uint64_t operator""_MB(unsigned long long value)
{
    return value * 1000_KB;
}

/**
 * @brief Convert gigabytes to bytes.
 */
constexpr std::uint64_t operator""_GB(unsigned long long value)
{
    return value * 1000_MB;
}

/**
 * @brief Convert terabytes to bytes.
 */
constexpr std::uint64_t operator""_TB(unsigned long long value)
{
    return value * 1000_GB;
}

/**
 * @brief Convert kibibytes to bytes.
 */
constexpr std::uint64_t operator""_KiB(unsigned long long value)
{
    return value * 1024_B;
}

/**
 * @brief Convert mebibytes to bytes.
 */
constexpr std::uint64_t operator""_MiB(unsigned long long value)
{
    return value * 1024_KiB;
}

/**
 * @brief Convert gibibytes to bytes.
 */
constexpr std::uint64_t operator""_GiB(unsigned long long value)
{
    return value * 1024_MiB;
}

/**
 * @brief Convert tebibytes to bytes.
 */
constexpr std::uint64_t operator""_TiB(unsigned long long value)
{
    return value * 1024_GiB;
}

/** @brief Default memory-cache retention budget. */
constexpr std::uint64_t default_memory_bytes = 256_MiB;

/** @brief Default persistent-cache retention budget. */
constexpr std::uint64_t default_disk_bytes = 4_GiB;

/**
 * @brief Immutable bytes for a serialized package, LTO fragment, PTX, or cubin.
 *
 * Storage preserves the exact byte count, including embedded NULs. No terminator
 * is appended. A blob keeps this storage alive independently of cache eviction
 * and of the artifact or kernel instance that produced it.
 */
class binary
{
    /**
     * @brief Owned byte storage; published views never modify or resize it.
     */
    std::vector<char> bytes_;

    /**
     * @brief Precomputed content identity, avoiding repeated hashing of large blobs on cache hits.
     */
    std::string digest_;

  public:
    /**
     * @brief Take ownership of bytes; pass an rvalue to transfer the vector's allocation.
     */
    explicit binary(std::vector<char> bytes);

    /**
     * @brief Borrow the exact stored range; valid while this binary remains alive.
     */
    std::span<char const> bytes() const;

    /**
     * @brief Alias of bytes(), with the same lifetime and byte count.
     */
    std::span<char const> to_view() const;

    /** @brief Borrow the content identity computed when these immutable bytes were created. */
    std::string_view content_digest() const;
};

/**
 * @brief Shared ownership of immutable bytes; copying the pointer does not copy the bytes.
 */
using blob = std::shared_ptr<binary const>;

/**
 * @brief Copy a borrowed byte range into independent immutable storage.
 * @param bytes Range copied in full, including any NUL bytes; may be empty.
 * @return An owning, non-null blob. The input can be released after the call.
 */
blob to_blob(std::span<char const> bytes);

/**
 * @brief Build immutable storage from an owning vector.
 * @param bytes Vector moved into storage; lvalues are copied into this parameter.
 * @return An owning, non-null blob, independent of the original vector's lifetime.
 */
blob to_blob(std::vector<char> bytes);

/**
 * @brief Compilation/link failures carry the complete backend diagnostics.
 */
class error : public std::runtime_error
{
  public:
    /**
     * @brief Status code preserved at the C API boundary.
     */
    clangrtcStatus status;

    /**
     * @brief Failing stage, such as request, artifact, compile, or nvjitlink.
     */
    std::string stage;

    /**
     * @brief Construct an error using the default status for its diagnostic stage.
     * @param stage Producing stage copied into the error.
     * @param message Owned diagnostic text.
     */
    error(std::string stage, std::string message);

    /**
     * @brief Construct an error preserving an explicit API status and diagnostic stage.
     * @param stage Producing stage copied into the error.
     * @param message Owned diagnostic text.
     * @param status Failure category returned at the C boundary.
     */
    error(std::string stage, std::string message, clangrtcStatus status);
};

/**
 * @brief Virtual header whose path and contents are owned by the request.
 */
struct header
{
    /**
     * @brief Virtual header filename used by include resolution.
     */
    std::string path;

    /**
     * @brief Complete header source text, copied into the compilation filesystem.
     */
    std::string contents;
};

/**
 * @brief CUDA source and ordered compiler options, borrowed by workflow calls.
 */
struct source
{
    /**
     * @brief Virtual filename used for diagnostics and include resolution.
     */
    std::string name = "/__curtc__/input.cu";

    /**
     * @brief Complete CUDA translation unit or consumer source text.
     */
    std::string text;

    /**
     * @brief Additional virtual headers made visible to this compilation.
     */
    std::vector<header> headers;

    /**
     * @brief Ordered clangrtc options, including repeated Clang/NVVM pass-through options.
     */
    std::vector<std::string> options;

    /**
     * @brief Optional expressions selecting definitions to retain; empty retains visible definitions.
     */
    std::vector<std::string> expressions;
};

/**
 * @brief Cache policy copied into each independently owned cache instance.
 */
struct configuration
{
    /**
     * @brief Preferred cache root, with fallback through XDG_CACHE_HOME, HOME, TMPDIR, and /tmp.
     *
     * CURTC_CACHE_DIR supplies the preference when this field is absent.
     * Unusable candidates are skipped; exhausting all candidates disables disk caching.
     */
    std::optional<std::filesystem::path> cacheDirectory;

    /**
     * @brief Budget for ready in-memory results; excludes in-flight work and caller-held blobs.
     */
    std::uint64_t memoryBytes = default_memory_bytes;

    /**
     * @brief Budget for published disk entries; temporary files can exceed this during publication.
     */
    std::uint64_t diskBytes = default_disk_bytes;

    /**
     * @brief Retain ready results; concurrent requests still share in-flight work if false.
     */
    bool memoryCache = true;

    /**
     * @brief Read and publish disk entries; failures fall back to compilation.
     */
    bool diskCache = true;

    /**
     * @brief Called outside cache locks; must not throw. Disk failures do not fail compilation.
     */
    std::function<void(std::string_view)> diagnostic;
};

/**
 * @brief Cumulative counters for one cache; concurrent activity may advance fields during the snapshot.
 */
struct cache_statistics
{
    /**
     * @brief Ready results reused from memory.
     */
    std::uint64_t memoryHits{};

    /**
     * @brief Validated disk results reused.
     */
    std::uint64_t diskHits{};

    /**
     * @brief Requests requiring computation after cache lookup.
     */
    std::uint64_t misses{};

    /**
     * @brief Requests waiting for an identical in-flight operation.
     */
    std::uint64_t sharedRequests{};

    /**
     * @brief Disk failures reported, including corrupt entries.
     */
    std::uint64_t diskErrors{};

    /**
     * @brief Kernel backend compilations attempted.
     */
    std::uint64_t kernelCompilations{};

    /**
     * @brief Source-fragment backend compilations attempted; excludes scans.
     */
    std::uint64_t fragmentCompilations{};

    /**
     * @brief nvJitLink finalizations attempted; excludes stored cubin retrieval.
     */
    std::uint64_t links{};
};

inline binary::binary(std::vector<char> bytes) : bytes_(std::move(bytes)), digest_(digest(bytes_))
{
}

inline std::span<char const> binary::bytes() const
{
    return bytes_;
}

inline std::span<char const> binary::to_view() const
{
    return bytes_;
}

inline std::string_view binary::content_digest() const
{
    return digest_;
}

inline blob to_blob(std::span<char const> bytes)
{
    return to_blob(std::vector<char>(bytes.begin(), bytes.end()));
}

inline blob to_blob(std::vector<char> bytes)
{
    return std::make_shared<binary const>(std::move(bytes));
}

/**
 * @brief Copy nonempty precompiled input into shared binary storage.
 * @param input Valid borrowed byte range.
 * @return Owned binary payload.
 * @throws std::invalid_argument If the input is empty.
 */
inline blob to_blob(clangrtcString input)
{
    CR_EXPECTS(input.size, "Missing precompiled input bytes", std::invalid_argument);
    return to_blob(std::span<char const>(input.data, input.size));
}

/**
 * @brief Map a diagnostic stage to its default API failure status.
 * @param stage Producing diagnostic stage.
 * @return Default API failure category for that stage.
 */
constexpr clangrtcStatus to_status(std::string_view stage)
{
    if (stage == "request")
    {
        return CLANGRTC_INVALID_ARGUMENT;
    }

    if (stage == "artifact")
    {
        return CLANGRTC_INVALID_ARTIFACT;
    }

    if (stage == "nvjitlink")
    {
        return CLANGRTC_LINK_FAILED;
    }

    return CLANGRTC_COMPILATION_FAILED;
}

inline error::error(std::string where, std::string message) : error(where, std::move(message), to_status(where))
{
}

inline error::error(std::string where, std::string message, clangrtcStatus code)
    : std::runtime_error(std::move(message)), status(code), stage(std::move(where))
{
}

/** @brief Maximum number of list entries decoded from a cached record. */
constexpr std::size_t max_record_entries = 65'536;

/**
 * @brief Internal encoding for cached code and symbol records.
 *
 * Integers and lengths are unsigned 64-bit little-endian values. Text and bytes
 * share length-delimited framing; only text is decoded into std::string.
 * Changing a record layout requires updating its enclosing schema identity.
 */
struct writer
{
    /**
     * @brief Owned serialized bytes; write operations append to this buffer.
     */
    std::vector<char> data;

    /**
     * @brief Append an unsigned integer in the cache format's little-endian encoding.
     * @param n Integer to serialize.
     */
    void number(std::uint64_t n)
    {
        for (int i = 0; i < 8; ++i)
        {
            data.push_back(static_cast<char>(n & 255));
            n >>= 8;
        }
    }

    /**
     * @brief Append a byte count followed by the exact payload, without a terminator.
     * @param s Bytes copied into the output record.
     */
    void bytes(std::span<char const> s)
    {
        number(s.size());
        data.insert(data.end(), s.begin(), s.end());
    }

    /**
     * @brief Append a string using the same length-delimited encoding as binary data.
     * @param s Text copied without scanning for a terminator.
     */
    void text(std::string_view s)
    {
        bytes({s.data(), s.size()});
    }

    /**
     * @brief Append an ordered string list with its element count.
     * @param xs Strings copied in their existing order.
     */
    void strings(std::span<std::string const> xs)
    {
        number(xs.size());

        for (auto const &x : xs)
        {
            text(x);
        }
    }
};

/**
 * @brief Checked cursor over a borrowed record; returned byte spans borrow the same storage.
 */
struct reader
{
    /**
     * @brief Remaining record bytes; reads advance this borrowed view.
     *
     * The backing storage must outlive the reader and any byte spans returned by it.
     */
    std::span<char const> data;

    /**
     * @brief Consume one little-endian unsigned integer.
     * @return Decoded integer.
     * @throws error If the record does not contain a complete integer.
     */
    std::uint64_t number()
    {
        if (data.size() < 8)
        {
            throw error("artifact", "Truncated integer");
        }

        std::uint64_t n{};

        for (int i = 0; i < 8; ++i)
        {
            n |= static_cast<std::uint64_t>(static_cast<unsigned char>(data[i])) << (8 * i);
        }

        data = data.subspan(8);
        return n;
    }

    /**
     * @brief Consume a length-delimited payload after checking its bounds.
     * @return Byte span borrowing the input record.
     * @throws error If the encoded payload is truncated.
     */
    std::span<char const> bytes()
    {
        auto n = number();

        if (n > data.size())
        {
            throw error("artifact", "Truncated section");
        }

        auto s = data.first(n);
        data = data.subspan(n);
        return s;
    }

    /**
     * @brief Consume a length-delimited string and copy it into owned storage.
     * @return Decoded string, preserving embedded NULs.
     */
    std::string text()
    {
        auto s = bytes();
        return {s.begin(), s.end()};
    }

    /**
     * @brief Consume a list count and validate it against format and remaining-data limits.
     * @return Validated element count.
     * @throws error If the list exceeds the decoding limit or cannot fit in the remaining record.
     */
    std::size_t count()
    {
        auto n = number();

        // Every element needs at least its length word; cap counts before iteration/allocation.
        if (n > max_record_entries || n > data.size() / sizeof(std::uint64_t))
        {
            throw error("artifact", "Invalid section count");
        }

        return n;
    }

    /**
     * @brief Consume an ordered list of length-delimited strings.
     * @return Owned strings in encoded order.
     */
    std::vector<std::string> strings()
    {
        std::vector<std::string> xs;
        auto n = count();

        for (size_t i = 0; i < n; ++i)
        {
            xs.push_back(text());
        }

        return xs;
    }

    /**
     * @brief Require the record to be fully consumed.
     * @throws error If trailing bytes remain.
     */
    void end()
    {
        if (!data.empty())
        {
            throw error("artifact", "Trailing bytes");
        }
    }
};

/**
 * @brief Canonical width of an XXH3-128 content digest.
 */
constexpr std::size_t digest_hex_size = content_hash_hex_size;

/**
 * @brief Delimiter between the encoded digest and payload.
 */
constexpr char digest_delimiter = '\n';

/**
 * @brief Frame a payload with schema magic, a hexadecimal checksum, and its delimiter.
 * @param magic Format marker copied before the checksum.
 * @param body Exact payload used to compute the checksum.
 * @return Owned framed bytes.
 */
inline std::vector<char> envelope(std::string_view magic, std::span<char const> body)
{
    std::vector<char> bytes(magic.begin(), magic.end());
    auto hash = digest(body);
    bytes.insert(bytes.end(), hash.begin(), hash.end());
    bytes.push_back(digest_delimiter);
    bytes.insert(bytes.end(), body.begin(), body.end());
    return bytes;
}

/**
 * @brief Validate framing and integrity before exposing a borrowed payload.
 * @param bytes Complete framed record.
 * @param magic Expected format marker.
 * @return Payload borrowing bytes; checksum validation does not authenticate its producer.
 * @throws error If the header or checksum is invalid.
 */
inline std::span<char const> unpack(std::span<char const> bytes, std::string_view magic)
{
    auto const digest_end = magic.size() + digest_hex_size;
    auto const header_size = digest_end + sizeof(digest_delimiter);

    if (bytes.size() < header_size || !std::ranges::equal(bytes.first(magic.size()), magic))
    {
        throw error("artifact", "Invalid package header");
    }

    auto body = bytes.subspan(header_size);
    std::string_view hash(bytes.data() + magic.size(), digest_hex_size);

    if (bytes[digest_end] != digest_delimiter || hash != digest(body))
    {
        throw error("artifact", "Package integrity check failed");
    }

    return body;
}

/**
 * @brief Read a file into owned binary storage.
 * @param path File to open in binary mode.
 * @return Exact file contents.
 * @throws std::runtime_error If opening or reading the file fails.
 */
inline std::vector<char> read_file(std::filesystem::path const &path)
{
    std::ifstream in(path, std::ios::binary);

    if (!in)
    {
        throw std::runtime_error(std::format("Cannot read {}", path.string()));
    }

    std::vector<char> data{std::istreambuf_iterator<char>(in), {}};

    if (in.bad())
    {
        throw std::runtime_error(std::format("Cannot read {}", path.string()));
    }

    return data;
}

/**
 * @brief Atomic event counters used for cache-wide totals.
 */
struct counters
{
    std::atomic<std::uint64_t> memoryHits{};

    std::atomic<std::uint64_t> diskHits{};

    std::atomic<std::uint64_t> misses{};

    std::atomic<std::uint64_t> sharedRequests{};

    std::atomic<std::uint64_t> diskErrors{};

    std::atomic<std::uint64_t> kernelCompilations{};

    std::atomic<std::uint64_t> fragmentCompilations{};

    std::atomic<std::uint64_t> links{};

    /**
     * @brief Snapshot the counters; concurrent updates can occur between field reads.
     * @return Current counter values.
     */
    cache_statistics get() const
    {
        return {memoryHits.load(),           diskHits.load(),   misses.load(),
                sharedRequests.load(),       diskErrors.load(), kernelCompilations.load(),
                fragmentCompilations.load(), links.load()};
    }
};

/**
 * @brief Environment lookup returning an owned value, or std::nullopt when unset.
 */
inline std::optional<std::string> get_env_var(std::string const &name)
{
    if (auto value = std::getenv(name.c_str()))
    {
        return std::string(value);
    }

    return std::nullopt;
}

/**
 * @brief Read an optional environment path, preserving present-but-empty values.
 */
inline std::optional<std::filesystem::path> get_envpath(
    std::string const &name, function_ref<std::optional<std::string>(std::string const &)> environment)
{
    if (auto value = environment(name))
    {
        return std::filesystem::path(*value);
    }

    return std::nullopt;
}

/**
 * @brief Discover ordered fallback cache roots without creating directories; temporary roots are per user.
 */
inline std::vector<std::filesystem::path> cache_directory_candidates(
    function_ref<std::optional<std::string>(std::string const &)> environment)
{
    std::vector<std::filesystem::path> directories;
    auto accessible = [](std::filesystem::path const &path) {
        std::error_code ec;
        return std::filesystem::is_directory(path, ec) && ::access(path.c_str(), R_OK | X_OK) == 0;
    };

    if (auto root = get_envpath("XDG_CACHE_HOME", environment))
    {
        directories.push_back(*root / "curtc");
    }

    if (auto root = get_envpath("HOME", environment); root && accessible(*root))
    {
        directories.push_back(*root / ".cache/curtc");
    }

    auto temporary = std::format("curtc-{}", ::getuid());

    if (auto root = get_envpath("TMPDIR", environment); root && accessible(*root))
    {
        directories.push_back(*root / temporary);
    }

    if (accessible("/tmp"))
    {
        directories.push_back(std::filesystem::path("/tmp") / temporary);
    }

    return directories;
}

/**
 * @brief Try the preferred root, then fallback candidates, selecting a usable versioned cache directory.
 *
 * CURTC_CACHE_DIR supplies the preference when no explicit root was given.
 * Probe actual file creation instead of trusting mode bits alone (for example,
 * on a read-only filesystem). A nullopt result disables disk caching. A probe
 * failure is reported but never prevents trying the remaining candidates.
 */
inline std::optional<std::filesystem::path> select_cache_directory(
    std::optional<std::filesystem::path> preferred, std::span<std::filesystem::path const> candidates,
    function_ref<std::optional<std::string>(std::string const &)> environment,
    function_ref<void(std::string_view)> report)
{
    if (!preferred)
    {
        preferred = get_envpath("CURTC_CACHE_DIR", environment);
    }

    std::vector<std::filesystem::path> directories;
    auto append = [&](std::filesystem::path const &root) {
        auto directory = root / "v1";

        if (std::find(directories.begin(), directories.end(), directory) == directories.end())
        {
            directories.push_back(std::move(directory));
        }
    };

    if (preferred)
    {
        append(*preferred);
    }

    for (auto const &candidate : candidates)
    {
        append(candidate);
    }

    for (auto const &directory : directories)
    {
        std::error_code ec;
        std::filesystem::create_directories(directory, ec);

        if (!ec && ::access(directory.c_str(), R_OK | W_OK | X_OK) != 0)
        {
            ec = std::error_code(errno, std::generic_category());
        }

        if (!ec)
        {
            auto probe = (directory / ".probe-XXXXXX").string();
            auto fd = ::mkstemp(probe.data());

            if (fd < 0)
            {
                ec = std::error_code(errno, std::generic_category());
            }
            else
            {
                if (::close(fd) != 0)
                {
                    ec = std::error_code(errno, std::generic_category());
                }

                std::error_code cleanup;
                std::filesystem::remove(probe, cleanup);

                if (!ec)
                {
                    ec = cleanup;
                }
            }
        }

        if (!ec)
        {
            return directory;
        }

        report(std::format("Cannot use disk cache {}: {}", directory.string(), ec.message()));
    }

    return std::nullopt;
}

/**
 * @brief Content-addressed stage results with bounded retention and shared in-flight work.
 *
 * The mutex protects ready/pending entries and LRU accounting. Compilation, disk
 * I/O, diagnostics, and waiting run outside it. Eviction drops only cache-owned
 * references; returned blobs retain their bytes independently.
 */
class content_cache
{
    configuration config_;

    /**
     * @brief Type-erased cached result paired with its memory-retention charge.
     */
    struct value
    {
        std::shared_ptr<void const> object;

        std::uint64_t size;
    };

    /**
     * @brief Ready memory-cache value with the recency tick used for eviction.
     */
    struct entry
    {
        value data;

        std::uint64_t tick;
    };

    std::mutex mutex_;

    std::once_flag directory_once_;

    std::map<std::string, entry> memory_;

    std::map<std::string, std::shared_future<value>> pending_;

    std::uint64_t bytes_{};

    std::uint64_t tick_{};

    /**
     * @brief Count a disk error and notify the optional warning callback outside cache locks.
     * @param message Diagnostic borrowed only for the callback.
     * @param local Optional counters for the request that encountered the error.
     * @note Callback exceptions are suppressed.
     */
    void report_disk_error(std::string_view message, curtcCompileStatistics *local)
    {
        count(stats.diskErrors, &curtcCompileStatistics::diskErrors, local);

        if (config_.diagnostic)
        {
            try
            {
                config_.diagnostic(message);
            }
            catch (...)
            {
            }
        }
    }

    /**
     * @brief Select a writable disk-cache location once, deferring warnings until locks are released.
     * @param local Optional request counters receiving disk-selection errors.
     */
    void initialize_directory(curtcCompileStatistics *local)
    {
        std::vector<std::string> warnings;
        std::call_once(directory_once_, [&] {
            if (config_.diskCache && config_.diskBytes)
            {
                auto directories = cache_directory_candidates(get_env_var);
                config_.cacheDirectory =
                    select_cache_directory(config_.cacheDirectory, directories, get_env_var,
                                           [&](std::string_view message) { warnings.emplace_back(message); });

                if (!config_.cacheDirectory)
                {
                    config_.diskCache = false;
                }
            }
        });

        // Callbacks may reenter the service; run them after both configuration and once locks are released.
        for (auto const &message : warnings)
        {
            report_disk_error(message, local);
        }
    }

    /**
     * @brief Retain a ready result within the memory budget while the caller holds the cache mutex.
     * @param key Content key copied into the memory map.
     * @param result Shared result and its retention charge.
     * @param retired Receives evicted ownership for destruction after the caller releases the mutex.
     */
    void retain_in_memory(std::string_view key, value const &result, std::vector<value> &retired)
    {
        if (!config_.memoryCache || result.size > config_.memoryBytes)
        {
            return;
        }

        while (result.size > config_.memoryBytes - bytes_ && !memory_.empty())
        {
            auto oldest = std::min_element(memory_.begin(), memory_.end(),
                                           [](auto const &a, auto const &b) { return a.second.tick < b.second.tick; });
            auto size = oldest->second.data.size;
            retired.push_back(std::move(oldest->second.data));
            bytes_ -= size;
            memory_.erase(oldest);
        }

        memory_.emplace(key, entry{result, ++tick_});
        bytes_ += result.size;
    }

    /**
     * @brief Read and validate a persisted stage result, treating failures as cache misses.
     * @param key Content key identifying the cache file.
     * @param local Optional counters receiving disk-read errors.
     * @return Owned payload, or an empty pointer when reuse is unavailable.
     */
    blob read_from_disk(std::string_view key, curtcCompileStatistics *local)
    {
        if (!config_.diskCache || !config_.diskBytes)
        {
            return {};
        }

        auto path = *config_.cacheDirectory / std::format("{}{}", key, file_extension);
        std::error_code ec;

        if (!std::filesystem::exists(path, ec))
        {
            if (ec)
            {
                report_disk_error(ec.message(), local);
            }

            return {};
        }

        try
        {
            auto file = read_file(path);
            auto body = unpack(file, format_magic);
            reader r{body};

            if (r.text() != key)
            {
                throw std::runtime_error("Wrong cache key");
            }

            // The cursor borrows file, so copy the payload before the local file buffer dies.
            auto result = to_blob(r.bytes());
            r.end();
            // Refresh the disk entry's age for approximate LRU eviction. Timestamp-update failures
            // only affect eviction order, so preserve the cache hit even when this fails.
            std::filesystem::last_write_time(path, std::filesystem::file_time_type::clock::now(), ec);
            return result;
        }
        catch (std::exception const &e)
        {
            report_disk_error(e.what(), local);
            std::filesystem::remove(path, ec);
            return {};
        }
    }

    /**
     * @brief Publish a result atomically and prune oldest disk entries to the retention budget.
     * @param key Content key encoded into the file and filename.
     * @param value Shared payload to persist.
     * @param local Optional counters receiving disk-write or pruning errors.
     * @note Disk failures are reported without failing compilation.
     */
    void write_to_disk(std::string_view key, blob const &value, curtcCompileStatistics *local)
    {
        if (!config_.diskCache || !config_.diskBytes)
        {
            return;
        }

        std::filesystem::path temporary;
        int fd = -1;

        try
        {
            writer w;
            w.text(key);
            w.bytes(value->bytes());
            auto bytes = envelope(format_magic, w.data);

            if (bytes.size() > config_.diskBytes)
            {
                return;
            }

            std::filesystem::create_directories(*config_.cacheDirectory);
            auto name = (*config_.cacheDirectory / ".write-XXXXXX").string();
            fd = ::mkstemp(name.data());

            if (fd < 0)
            {
                throw std::runtime_error("Cannot create cache temporary file");
            }

            temporary = name;
            size_t offset{};

            while (offset < bytes.size())
            {
                auto n = ::write(fd, bytes.data() + offset, bytes.size() - offset);

                if (n < 0 && errno == EINTR)
                {
                    continue;
                }

                if (n <= 0)
                {
                    throw std::runtime_error("Cannot write cache entry");
                }

                offset += n;
            }

            if (::fsync(fd) != 0)
            {
                throw std::runtime_error("Cannot sync cache entry");
            }

            auto result = ::close(fd);
            fd = -1;

            if (result != 0)
            {
                throw std::runtime_error("Cannot close cache entry");
            }

            // Publish in one rename so other processes never observe a partially written entry.
            std::filesystem::rename(temporary, *config_.cacheDirectory / std::format("{}{}", key, file_extension));
            temporary.clear();
            std::vector<std::pair<std::filesystem::file_time_type, std::filesystem::path>> files;
            std::uint64_t total{};

            for (auto const &f : std::filesystem::directory_iterator(*config_.cacheDirectory))
            {
                auto filename = f.path().filename().string();

                if (filename.size() != key.size() + file_extension.size() || !filename.ends_with(file_extension) ||
                    !f.is_regular_file())
                {
                    continue;
                }

                total += f.file_size();
                files.emplace_back(f.last_write_time(), f.path());
            }

            std::sort(files.begin(), files.end());

            for (auto const &[time, path] : files)
            {
                if (total <= config_.diskBytes)
                {
                    break;
                }

                auto size = std::filesystem::file_size(path);
                std::filesystem::remove(path);
                total -= size;
            }
        }
        catch (std::exception const &e)
        {
            if (fd >= 0)
            {
                ::close(fd);
            }

            if (!temporary.empty())
            {
                std::error_code ec;
                std::filesystem::remove(temporary, ec);
            }

            report_disk_error(e.what(), local);
        }
    }

    /**
     * @brief Share one computation per stage key and retain its ready result according to cache policy.
     * @param stage Namespace separating independent cache stages.
     * @param inputs Ordered bytes defining the stage inputs.
     * @param make Work executed outside the cache mutex after a miss.
     * @param local Optional counters attributed to this request.
     * @return Shared result, including when another request performed the work.
     * @note Failed work is removed from the pending map so later requests can retry.
     */
    value get_cached(std::string_view stage, std::span<char const> inputs, function_ref<value(std::string_view)> make,
                     curtcCompileStatistics *local)
    {
        initialize_directory(local);
        writer identity;
        identity.text(stage);
        identity.bytes(inputs);
        auto key = digest(identity.data);
        std::shared_future<value> future;
        std::shared_ptr<std::promise<value>> promise;
        {
            std::lock_guard lock(mutex_);

            if (auto it = memory_.find(key); it != memory_.end())
            {
                count(stats.memoryHits, &curtcCompileStatistics::memoryHits, local);
                it->second.tick = ++tick_;
                return it->second.data;
            }

            if (auto it = pending_.find(key); it != pending_.end())
            {
                count(stats.sharedRequests, &curtcCompileStatistics::sharedRequests, local);
                future = it->second;
            }
            else
            {
                promise = std::make_shared<std::promise<value>>();
                future = promise->get_future().share();
                pending_.emplace(key, future);
            }
        }

        if (!promise)
        {
            return future.get();
        }

        try
        {
            auto result = make(key);
            std::vector<value> retired;
            {
                std::lock_guard lock(mutex_);
                retain_in_memory(key, result, retired);
                promise->set_value(result);
                pending_.erase(key);
            }

            // Loaded CUDA libraries can be released here; never unload them while holding the cache mutex.
            return result;
        }
        catch (...)
        {
            std::lock_guard lock(mutex_);
            promise->set_exception(std::current_exception());
            pending_.erase(key);
            throw;
        }
    }

  public:
    counters stats;

    /**
     * @brief Increment both the cache-wide event count and an optional request-local count.
     * @param total Shared atomic counter selected for the event.
     * @param field Matching request-local field.
     * @param local Optional statistics owned by the current request.
     */
    void count(std::atomic<std::uint64_t> &total, uint64_t curtcCompileStatistics::*field,
               curtcCompileStatistics *local)
    {
        ++total;

        if (local)
        {
            ++(local->*field);
        }
    }

    /**
     * @brief Copy cache policy without probing directories or loading a CUDA driver.
     * @param config Independent configuration owned by this cache.
     */
    explicit content_cache(configuration config) : config_(std::move(config))
    {
    }

    /**
     * @brief Retrieve or persist a binary stage result without collecting request-local statistics.
     * @param stage Namespace identifying the compilation or linking stage.
     * @param inputs Ordered bytes defining the stage key.
     * @param make Factory invoked after a cache miss.
     * @return Shared binary result.
     */
    blob get(std::string_view stage, std::span<char const> inputs, function_ref<blob()> make)
    {
        return get(stage, inputs, make, nullptr);
    }

    /**
     * @brief Retrieve or persist a binary stage result and attribute events to its request.
     * @param stage Namespace identifying the compilation or linking stage.
     * @param inputs Ordered bytes defining the stage key.
     * @param make Factory invoked after a cache miss.
     * @param local Optional request counters.
     * @return Shared result reused from memory, disk, concurrent work, or the factory.
     */
    blob get(std::string_view stage, std::span<char const> inputs, function_ref<blob()> make,
             curtcCompileStatistics *local)
    {
        auto result = get_cached(
            stage, inputs,
            [&](std::string_view key) {
                auto result = read_from_disk(key, local);

                if (result)
                {
                    count(stats.diskHits, &curtcCompileStatistics::diskHits, local);
                }
                else
                {
                    count(stats.misses, &curtcCompileStatistics::misses, local);
                    result = make();

                    if (!result)
                    {
                        throw std::runtime_error("Empty compilation result");
                    }

                    write_to_disk(key, result, local);
                }

                return value{result, result->bytes().size()};
            },
            local);
        return std::static_pointer_cast<binary const>(result.object);
    }

    /**
     * @brief Share an immutable object in memory without serializing it to disk.
     * @tparam T Owned object type.
     * @param stage Namespace identifying the object stage.
     * @param inputs Ordered bytes defining the object key.
     * @param size Estimated retention cost charged to the memory budget.
     * @param make Factory invoked outside locks after a miss.
     * @param local Optional request counters.
     * @return Shared immutable object.
     */
    template <class T>
    std::shared_ptr<T const> get_object(std::string_view stage, std::span<char const> inputs, size_t size,
                                        function_ref<std::shared_ptr<T const>()> make, curtcCompileStatistics *local)
    {
        auto result = get_cached(
            stage, inputs,
            [&](std::string_view) {
                count(stats.misses, &curtcCompileStatistics::misses, local);
                auto object = make();

                if (!object)
                {
                    throw std::runtime_error("Empty cached object");
                }

                return value{std::move(object), size};
            },
            local);
        return std::static_pointer_cast<T const>(result.object);
    }
};

#ifndef CURTC_INTERNAL_TESTS

namespace
{

/**
 * @brief Deliver a borrowed diagnostic synchronously while suppressing callback exceptions.
 * @param callback Optional user callback.
 * @param user Context supplied with the callback.
 * @param level Severity of the diagnostic.
 * @param stage Producing compilation, linking, or loading stage.
 * @param message Text borrowed for the callback duration.
 */
void diagnostic(clangrtcDiagnosticCallback callback, void *user, clangrtcDiagnosticLevel level, std::string_view stage,
                std::string_view message) noexcept
{
    if (callback)
    {
        clangrtcDiagnostic info{level, {stage.data(), stage.size()}, {message.data(), message.size()}, {}, 0, 0};

        try
        {
            callback(&info, user);
        }
        catch (...)
        {
        }
    }
}

/**
 * @brief Translate failures at the C boundary; diagnostic views borrow caught exceptions.
 */
clangrtcStatus call(clangrtcDiagnosticCallback callback, void *user, function_ref<void()> work) noexcept
{
    try
    {
        work();
        return CLANGRTC_SUCCESS;
    }
    catch (error const &e)
    {
        diagnostic(callback, user, CLANGRTC_DIAGNOSTIC_ERROR, e.stage, e.what());
        return e.status;
    }
    catch (Failure const &e)
    {
        diagnostic(callback, user, CLANGRTC_DIAGNOSTIC_ERROR, e.stage, e.what());
        return e.status;
    }
    catch (std::bad_alloc const &)
    {
        diagnostic(callback, user, CLANGRTC_DIAGNOSTIC_ERROR, "allocation", "Host allocation failed");
        return CLANGRTC_OUT_OF_MEMORY;
    }
    catch (std::logic_error const &e)
    {
        diagnostic(callback, user, CLANGRTC_DIAGNOSTIC_ERROR, "request", e.what());
        return CLANGRTC_INVALID_ARGUMENT;
    }
    catch (std::exception const &e)
    {
        diagnostic(callback, user, CLANGRTC_DIAGNOSTIC_ERROR, "internal", e.what());
        return CLANGRTC_INTERNAL_ERROR;
    }
    catch (...)
    {
        diagnostic(callback, user, CLANGRTC_DIAGNOSTIC_ERROR, "internal", "Unknown workflow failure");
        return CLANGRTC_INTERNAL_ERROR;
    }
}

/**
 * @brief Copy a valid C string view into owned storage using its explicit byte count.
 * @param input Borrowed text or bytes.
 * @return Owned copy preserving embedded NULs.
 */
std::string to_string(clangrtcString input)
{
    return std::string(input.data, input.size);
}

/**
 * @brief Copy and validate a name that must not contain NUL bytes.
 * @param input Borrowed name with an explicit byte count.
 * @return Owned name suitable for a CUDA or filesystem boundary.
 * @throws std::invalid_argument If the view contains a NUL byte.
 */
std::string to_name(clangrtcString input)
{
    auto result = to_string(input);
    CR_EXPECTS(result.find('\0') == std::string::npos, "Name contains NUL bytes", std::invalid_argument);
    return result;
}

/**
 * @brief Copy an ordered array of NUL-free C string views.
 * @param data Valid array of borrowed views.
 * @param count Number of views to copy.
 * @return Owned strings in input order.
 */
std::vector<std::string> to_strings(clangrtcString const *data, size_t count)
{
    std::vector<std::string> result;

    for (auto input : std::span(data, count))
    {
        result.push_back(to_name(input));
    }

    return result;
}

/**
 * @brief Convert a C input into an owned source description, including headers, options, and expressions.
 * @param input CUDA source or artifact consumer input.
 * @return Owned source description.
 */
source to_source(curtcInput const &input)
{
    source result;
    auto path = to_name(input.source.source.path);

    if (!path.empty())
    {
        result.name = std::move(path);
    }

    result.text = to_string(input.source.source.data);

    for (auto const &header : std::span(input.source.headers, input.source.headerCount))
    {
        result.headers.push_back({to_name(header.path), to_string(header.data)});
    }

    result.options = to_strings(input.options, input.optionCount);
    result.expressions = to_strings(input.source.expressions, input.source.expressionCount);
    return result;
}

/**
 * @brief Dynamically resolved context-independent CUDA loading API.
 *
 * The process-wide driver context is assumed to outlive all loaded libraries.
 * No CUDA context is created here.
 */
class cu_context
{
    void *library_;

    /**
     * @brief Resolve a required CUDA driver entry point from the owned shared library.
     * @tparam F Function-pointer type of the CUDA API.
     * @param name NUL-terminated exported function name.
     * @return Typed driver function pointer.
     * @throws error If the symbol is unavailable.
     */
    template <class F> F symbol(char const *name)
    {
        auto result = reinterpret_cast<F>(::dlsym(library_, name));

        if (!result)
        {
            throw error("cuda", std::format("Missing CUDA driver symbol {}", name), CLANGRTC_TOOLCHAIN_ERROR);
        }

        return result;
    }

  public:
    CR_CUDA_ENTRIES(CR_DECLARE_API_ENTRY)

    /**
     * @brief Open the CUDA driver, resolve its entry points, and initialize it without creating a context.
     * @throws error If driver loading, symbol resolution, or initialization fails.
     * @note Constructor failures close the shared-library handle before propagating.
     */
    cu_context() : library_(::dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL))
    {
        if (!library_)
        {
            auto message = ::dlerror();
            throw error("cuda", message ? message : "Cannot load CUDA driver", CLANGRTC_TOOLCHAIN_ERROR);
        }

        try
        {
            CR_CUDA_ENTRIES(CR_LOAD_CUDA_ENTRY)

            CR_CHECK_CUDA(cuInit(0), cuGetErrorString);
        }
        catch (...)
        {
            ::dlclose(library_);
            throw;
        }
    }

    /**
     * @brief Disable copying of the uniquely owned shared-library handle.
     */
    cu_context(cu_context const &) = delete;

    /**
     * @brief Disable assignment that would duplicate ownership of the driver handle.
     */
    cu_context &operator=(cu_context const &) = delete;

    /**
     * @brief Close the owned CUDA driver shared-library handle.
     */
    ~cu_context()
    {
        ::dlclose(library_);
    }
};

/** @brief Own nvJitLink's C API, bound directly for static builds or resolved from the selected shared library. */
class nvJitLink_context
{
#if !CLANGRTC_STATIC_LINK_NVJITLINK
    /** @brief Owned shared-library handle, kept open while the dispatch table is in use. */
    void *library_{};

    /**
     * @brief Resolve a required nvJitLink entry point from the owned shared library.
     * @tparam F Function-pointer type matching the backend ABI.
     * @param name NUL-terminated external entry-point name.
     * @return Function pointer valid until this context is destroyed.
     * @throws Failure With toolchain-error status if the symbol is unavailable.
     */
    template <class F> F symbol(char const *name) const
    {
        auto result = reinterpret_cast<F>(::dlsym(library_, name));

        if (!result)
        {
            auto message = ::dlerror();
            throw Failure(
                CLANGRTC_TOOLCHAIN_ERROR,
                std::format("Missing backend symbol {}: {}", name, message ? message : "Unknown loader error"),
                "toolchain");
        }

        return result;
    }

#endif

  public:
    CR_NVJITLINK_ENTRIES(CR_DECLARE_API_ENTRY)

    /**
     * @brief Initialize nvJitLink's dispatch table using the selected static or shared backend.
     * @throws Failure With toolchain-error status if loading or symbol resolution fails.
     * @note Constructor failures close the shared-library handle before propagating.
     */
    nvJitLink_context()
    {
#if !CLANGRTC_STATIC_LINK_NVJITLINK
        constexpr char const *library_path = CLANGRTC_NVJITLINK_LIBRARY;

        library_ = ::dlopen(library_path, RTLD_NOW | RTLD_LOCAL);

        if (!library_)
        {
            library_ = ::dlopen(std::filesystem::path(library_path).filename().c_str(), RTLD_NOW | RTLD_LOCAL);
        }

        if (!library_)
        {
            auto message = ::dlerror();
            throw Failure(CLANGRTC_TOOLCHAIN_ERROR,
                          std::format("Cannot load {}: {}", library_path, message ? message : "Unknown loader error"),
                          "toolchain");
        }

#endif

        try
        {
            CR_NVJITLINK_ENTRIES(CR_LOAD_NVJITLINK_ENTRY)
        }
        catch (...)
        {
#if !CLANGRTC_STATIC_LINK_NVJITLINK
            ::dlclose(library_);
#endif

            throw;
        }
    }

    /** @brief Disable copying of the backend context and its owned handle. */
    nvJitLink_context(nvJitLink_context const &) = delete;

    /** @brief Disable assignment of the backend context and its owned handle. */
    nvJitLink_context &operator=(nvJitLink_context const &) = delete;

    /** @brief Close the owned nvJitLink shared library when using dynamic linking. */
    ~nvJitLink_context()
    {
#if !CLANGRTC_STATIC_LINK_NVJITLINK
        ::dlclose(library_);
#endif
    }
};

/** @brief Initialize the linker dispatch table once; failures leave it available for retry. */
nvJitLink_context const &nvJitLink()
{
    static nvJitLink_context const context;
    return context;
}

/** @brief Borrow the process-wide CUDA driver context, which must outlive all loaded libraries. */
cu_context const &cu()
{
    static cu_context const context;
    return context;
}

/**
 * @brief Compiler output bytes with resolved symbols and associated CUDA source names.
 *
 * Bytes contain a prepared artifact or device code, depending on the operation.
 */
struct compiler_output
{
    /** @brief Owned output bytes, shared independently of the compiler result or cache. */
    blob code;

    /** @brief Resolved symbols for selected expressions, in request order. */
    std::vector<std::string> symbols;

    /** @brief CUDA source names aligned with symbols; entries are empty when unavailable. */
    std::vector<std::string> cuda_names;
};

/**
 * @brief Validate an sm_NN target and its optional architecture suffix.
 * @param target Requested GPU architecture.
 * @throws std::invalid_argument If the target has an unsupported spelling.
 */
void check_target(std::string_view target)
{
    CR_EXPECTS(target.starts_with("sm_") && target.size() > 3,
               std::format("Expected GPU target sm_NN, got '{}'", target), std::invalid_argument);
    auto suffix = target.substr(3);

    if (suffix.ends_with("a") || suffix.ends_with("f"))
    {
        suffix.remove_suffix(1);
    }

    CR_EXPECTS(!suffix.empty() &&
                   std::all_of(suffix.begin(), suffix.end(), [](char c) { return c >= '0' && c <= '9'; }),
               std::format("Invalid GPU architecture in target '{}'", target), std::invalid_argument);
}

/**
 * @brief Reject options whose source mode, target, or output is controlled by curtc.
 * @param flags Ordered compiler options to validate.
 * @throws std::invalid_argument If an option conflicts with the workflow.
 */
void check_compiler_options(std::span<std::string const> flags)
{
    for (auto const &flag : flags)
    {
        CR_EXPECTS(flag != "--require-kernel" && flag != "--host-only" && !flag.starts_with("--source=") &&
                       !flag.starts_with("--artifact=") && !flag.starts_with("--gpu-architecture=") &&
                       !flag.starts_with("--cuda-gpu-arch=") && !flag.starts_with("--emit=") && flag != "--lto-ir" &&
                       flag != "--ptx-text",
                   std::format("Compiler option is controlled by curtc: {}", flag), std::invalid_argument);
    }
}

/**
 * @brief Build C string views borrowing a stable ordered string array.
 * @param values Strings that must outlive all uses of the returned views.
 * @return Borrowed views in input order.
 */
std::vector<clangrtcString> to_views(std::span<std::string const> values)
{
    std::vector<clangrtcString> result;

    for (auto const &value : values)
    {
        result.push_back({value.data(), value.size()});
    }

    return result;
}

/**
 * @brief Invoke clangrtc preparation or compilation and copy its output and symbol metadata.
 * @param input Source, virtual headers, and expressions borrowed for this call.
 * @param flags Ordered compiler options.
 * @param environment Prepared artifact borrowed for compilation, or NULL.
 * @param preparing Select artifact preparation instead of compilation.
 * @param local Request receiving this invocation's frontend counters and elapsed time, including failed work.
 * @return Owned output bytes and symbol metadata, independent of the clangrtc result handle.
 * @throws error If clangrtc fails, preserving its status and diagnostics.
 */
compiler_output invoke_clangrtc(source const &input, std::span<std::string const> flags, clangrtcArtifact environment,
                                bool preparing, curtcCompileStatistics &local)
{
    std::vector<clangrtcFile> headers;

    for (auto const &header : input.headers)
    {
        headers.push_back({{header.path.data(), header.path.size()}, {header.contents.data(), header.contents.size()}});
    }

    auto expressions = to_views(input.expressions);
    auto options = to_views(flags);
    clangrtcRequest request{};
    request.source = {{input.name.data(), input.name.size()}, {input.text.data(), input.text.size()}};
    request.headers = headers.data();
    request.headerCount = headers.size();
    request.expressions = expressions.data();
    request.expressionCount = expressions.size();
    request.artifact = environment;
    clangrtcResult handle{};
    CR_DEFER([&] { clangrtcDestroyResult(handle); });
    auto status = preparing ? clangrtcPrepare(&request, options.data(), options.size(), nullptr, nullptr, &handle)
                            : clangrtcCompile(&request, options.data(), options.size(), nullptr, nullptr, &handle);
    clangrtcResultView info{};
    clangrtcGetResult(handle, &info);
    local.frontendNs += info.frontendNs;

    if (status != CLANGRTC_SUCCESS)
    {
        throw error("compile", to_string(info.log), status);
    }

    compiler_output result{to_blob(std::span<char const>(info.output.data, info.output.size)), {}, {}};

    for (size_t i = 0; i < info.symbolCount; ++i)
    {
        result.symbols.push_back(to_string(info.symbols[i].name));
        result.cuda_names.push_back(to_string(info.symbols[i].cudaName));
    }

    return result;
}

/**
 * @brief Owns a loaded clangrtc artifact and its borrowed metadata views.
 */
struct artifact
{
    clangrtcArtifact handle{};

    clangrtcArtifactInfo info{};

    /**
     * @brief Load a serialized artifact and inspect its immutable metadata.
     * @param data Serialized clangrtc artifact bytes.
     * @throws error If loading the artifact fails.
     */
    explicit artifact(std::span<char const> data)
    {
        auto status = clangrtcLoadArtifact({data.data(), data.size()}, nullptr, nullptr, &handle);

        if (status != CLANGRTC_SUCCESS)
        {
            throw error("artifact", "Cannot load serialized artifact", status);
        }

        clangrtcGetArtifactInfo(handle, &info);
    }

    /**
     * @brief Release the loaded clangrtc artifact and invalidate its borrowed metadata views.
     */
    ~artifact()
    {
        clangrtcDestroyArtifact(handle);
    }
};

/**
 * @brief Build the consumer source and compiler options used to reuse a prepared artifact.
 *
 * Inherit preparation options except output selectors, since curtc selects LTO for the consumer.
 * @param consumer Original request source, headers, options, and expressions, transferred into this call.
 * @param origin CURTC_INPUT_TYPE_CUDA for an internal source snapshot, or CURTC_INPUT_TYPE_ARTIFACT for a
 * caller-provided artifact.
 * @param prepared Metadata borrowed from the loaded artifact until this call returns.
 * @return Owned consumer with inherited options and any required synthetic source.
 */
source make_artifact_consumer(source consumer, curtcInputType origin, clangrtcArtifactInfo const &prepared)
{
    auto prepared_options = to_strings(prepared.options, prepared.optionCount);
    std::vector<std::string> inherited_options;
    bool option_argument = false;

    // Preparation output does not select consumer output. Preserve option values
    // verbatim, even when an argument happens to look like an output selector.
    for (auto const &option : prepared_options)
    {
        if (!option_argument && (option == "--lto-ir" || option == "--ptx-text" || option.starts_with("--emit=")))
        {
            continue;
        }

        inherited_options.push_back(option);
        option_argument =
            !option_argument && (option == "--clang-option" || option == "--nvvm-option" || option == "-I" ||
                                 option == "-isystem" || option == "-include" || option == "-D" || option == "-U");
    }

    if (origin == CURTC_INPUT_TYPE_CUDA)
    {
        // The snapshot already contains the source and its options. Empty source tells clangrtc
        // to compile that preprocessed payload instead of appending the original source again.
        consumer.text.clear();
        consumer.options = std::move(inherited_options);
        // A distinct virtual filename avoids replacing a source dependency stored in the artifact.
        consumer.name = "/__curtc__/snapshot.cu";
        return consumer;
    }

    // Caller-provided artifacts supply the base options; consumer options follow them in request order.
    inherited_options.insert(inherited_options.end(), consumer.options.begin(), consumer.options.end());
    consumer.options = std::move(inherited_options);

    if (!consumer.text.empty())
    {
        return consumer;
    }

    consumer.name = "/__curtc__/consumer.cu";
    auto module = to_string(prepared.moduleName);

    if (module.empty())
    {
        // PCH declarations are loaded implicitly, but clangrtc still requires nonempty consumer source.
        consumer.text = "\n";
        return consumer;
    }

    // BMI declarations become visible to expression instantiation only after an explicit import.
    consumer.text = std::format("import {};\n", module);
    return consumer;
}

/**
 * @brief Stable stage identity from the cache schema and build-time toolchain fingerprint.
 */
std::vector<char> identity(std::string_view stage)
{
    writer writer;
    writer.text(cache_identity_version);
    writer.text(CURTC_BUILD_ID);
    writer.text(stage);
    return writer.data;
}

/**
 * @brief Retrieve cached LTO and symbol metadata, compiling source or an artifact consumer on a miss.
 *
 * Source dependencies are snapshotted before cache lookup so changed headers invalidate the result.
 * Prepared inputs borrow an already loaded immutable artifact and its precomputed identity.
 * @param cache Cache retaining compiled LTO and its symbol metadata.
 * @param value CUDA source or a ready artifact with optional consumer source.
 * @param entries Selected kernel expressions compiled together.
 * @param target GPU architecture used for compilation and cache identity.
 * @param local Counters for the current request.
 * @return LTO bytes and selected symbol metadata, owned independently of cache eviction.
 */
compiler_output get_or_compile_lto(content_cache &cache, curtcInput const &value, std::span<std::string const> entries,
                                   std::string_view target, curtcCompileStatistics &local)
{
    auto input = to_source(value);
    check_compiler_options(input.options);

    auto retained_count = input.expressions.size();
    input.expressions.insert(input.expressions.end(), entries.begin(), entries.end());
    bool kernel = !entries.empty();
    blob snapshot;
    clangrtcArtifactInfo prepared{};
    std::string_view package_identity;

    if (value.type == CURTC_INPUT_TYPE_ARTIFACT)
    {
        CR_EXPECTS(value.source.artifact, "Missing prepared artifact handle", std::invalid_argument);
        clangrtcGetArtifactInfo(value.source.artifact, &prepared);
        package_identity = {prepared.contentHash.data, prepared.contentHash.size};
    }
    else
    {
        auto flags = input.options;
        flags.insert(flags.end(), {std::format("--gpu-architecture={}", target), "--artifact=preprocessed-cuda"});
        auto scan = input;
        scan.expressions.clear();
        snapshot = invoke_clangrtc(scan, flags, nullptr, true, local).code;
        package_identity = read_artifact_envelope(snapshot->bytes()).checksum;
    }

    writer key;
    key.bytes(identity(kernel ? "kernels" : "fragment"));
    key.text(artifact_version);
    key.text(package_identity);
    key.text(target);
    key.text(input.name);
    key.text(input.text);
    key.strings(input.options);
    key.strings(input.expressions);
    key.number(retained_count);
    key.number(input.headers.size());

    for (auto const &header : input.headers)
    {
        key.text(header.path);
        key.text(header.contents);
    }

    auto record = cache.get(
        kernel ? "kernels" : "fragment", key.data,
        [&] {
            if (kernel)
            {
                cache.count(cache.stats.kernelCompilations, &curtcCompileStatistics::kernelCompilations, &local);
            }
            else
            {
                cache.count(cache.stats.fragmentCompilations, &curtcCompileStatistics::fragmentCompilations, &local);
            }

            std::unique_ptr<artifact> snapshot_environment;
            auto environment = value.source.artifact;

            if (snapshot)
            {
                snapshot_environment = std::make_unique<artifact>(snapshot->bytes());
                environment = snapshot_environment->handle;
                prepared = snapshot_environment->info;
            }

            CR_EXPECTS(to_string(prepared.target) == target,
                       std::format("Artifact target '{}' does not match request target '{}'",
                                   to_string(prepared.target), target),
                       std::invalid_argument);
            auto consumer = make_artifact_consumer(std::move(input), value.type, prepared);
            consumer.options.push_back("--lto-ir");

            auto compiled = invoke_clangrtc(consumer, consumer.options, environment, false, local);
            CR_EXPECTS(compiled.symbols.size() == retained_count + entries.size() &&
                           compiled.cuda_names.size() == compiled.symbols.size(),
                       "Compiler symbols do not match the requested expressions", std::invalid_argument);

            // Retained definitions may include device functions and variables. Only the
            // explicit kernel entries become metadata, and each must resolve to a kernel.
            compiled.symbols.erase(compiled.symbols.begin(), compiled.symbols.begin() + retained_count);
            compiled.cuda_names.erase(compiled.cuda_names.begin(), compiled.cuda_names.begin() + retained_count);

            for (size_t i = 0; i < entries.size(); ++i)
            {
                CR_EXPECTS(!compiled.cuda_names[i].empty(),
                           std::format("Selected expression '{}' is not a CUDA kernel", entries[i]), Failure,
                           CLANGRTC_INVALID_EXPRESSION);
            }

            writer writer;
            writer.strings(compiled.symbols);
            writer.strings(compiled.cuda_names);
            writer.bytes(compiled.code->bytes());
            return to_blob(std::move(writer.data));
        },
        &local);
    reader reader{record->bytes()};
    auto symbols = reader.strings();
    auto cuda_names = reader.strings();
    auto code = to_blob(reader.bytes());
    reader.end();
    CR_EXPECTS(cuda_names.size() == symbols.size(), "Cached CUDA names do not match the symbol count",
               std::invalid_argument);

    CR_EXPECTS(symbols.size() == entries.size(), "Cached symbols do not match the requested entry count",
               std::invalid_argument);

    return {std::move(code), std::move(symbols), std::move(cuda_names)};
}

/**
 * @brief nvJitLink input representation paired with shared ownership of its bytes.
 */
struct link_input
{
    nvJitLinkInputType type;

    blob code;
};

/**
 * @brief Produce device code for loading or linking from any supported input format.
 *
 * CUDA source and artifact consumers use cached LTO compilation. Precompiled
 * LTO, PTX, and cubin inputs are copied with their selected binary symbols.
 * @param cache Cache used for source compilation.
 * @param input Source, artifact, LTO, PTX, or cubin input.
 * @param entries Requested kernel expressions or binary symbols.
 * @param target GPU architecture used for source compilation.
 * @param local Counters for the current request.
 * @return Code and any selected symbol metadata.
 * @throws std::invalid_argument If the input type is unsupported.
 */
compiler_output prepare_device_code(content_cache &cache, curtcInput const &input, std::span<std::string const> entries,
                                    std::string_view target, curtcCompileStatistics &local)
{
    switch (input.type)
    {
    case CURTC_INPUT_TYPE_CUDA:
    case CURTC_INPUT_TYPE_ARTIFACT:
        check_target(target);
        return get_or_compile_lto(cache, input, entries, target, local);

    case CURTC_INPUT_TYPE_LTO:
    case CURTC_INPUT_TYPE_PTX:
    case CURTC_INPUT_TYPE_CUBIN:
        return {to_blob(input.data), std::vector<std::string>(entries.begin(), entries.end()),
                std::vector<std::string>(entries.size())};

    default:
        CR_FAIL(std::format("Unsupported curtc input type: {}", static_cast<int>(input.type)), std::invalid_argument);
    }
}

/**
 * @brief Convert a curtc input type to the nvJitLink type of its prepared device code.
 * @param type Source/artifact inputs are represented by their compiled LTO IR.
 * @return Matching nvJitLink input tag.
 * @throws std::invalid_argument If the type is unsupported.
 */
nvJitLinkInputType to_nvJitLink_input_type(curtcInputType type)
{
    switch (type)
    {
    case CURTC_INPUT_TYPE_CUDA:
    case CURTC_INPUT_TYPE_ARTIFACT:
    case CURTC_INPUT_TYPE_LTO:
        return NVJITLINK_INPUT_LTOIR;

    case CURTC_INPUT_TYPE_PTX:
        return NVJITLINK_INPUT_PTX;

    case CURTC_INPUT_TYPE_CUBIN:
        return NVJITLINK_INPUT_CUBIN;

    default:
        CR_FAIL(std::format("Unsupported curtc input type: {}", static_cast<int>(type)), std::invalid_argument);
    }
}

/**
 * @brief Retrieve or finalize an ordered input combination as a cached cubin.
 * @param cache Cache used for finalized code.
 * @param inputs Ordered device-code fragments to link.
 * @param target GPU architecture passed to nvJitLink.
 * @param flags Additional linker options validated against workflow-owned options.
 * @param local Counters receiving link and cache events.
 * @return Shared finalized cubin.
 * @throws error If nvJitLink fails, preserving its diagnostic log.
 */
blob link(content_cache &cache, std::span<link_input const> inputs, std::string_view target,
          std::span<std::string const> flags, curtcCompileStatistics &local)
{
    check_target(target);
    CR_EXPECTS(!inputs.empty(), "At least one link input is required", std::invalid_argument);

    for (auto const &flag : flags)
    {
        auto name = flag.substr(0, flag.find('='));
        CR_EXPECTS(flag.starts_with("-") && name != "-arch" && name != "-ptx" && name != "-ltoir" && name != "-lto" &&
                       name != "-nvvm-latest",
                   std::format("Invalid or curtc-controlled nvJitLink option: {}", flag), std::invalid_argument);
    }

    writer key;
    key.bytes(identity("link"));
    key.text(target);
    key.strings(flags);

    for (auto const &input : inputs)
    {
        key.number(input.type);
        key.text(input.code->content_digest());
    }

    return cache.get(
        "cubin", key.data,
        [&] {
            cache.count(cache.stats.links, &curtcCompileStatistics::links, &local);
            auto const &api = nvJitLink();
            nvJitLinkHandle handle{};
            CR_DEFER([&] {
                if (handle)
                {
                    api.nvJitLinkDestroy(&handle);
                }
            });
            auto arch = std::format("-arch={}", target);
            std::vector<char const *> options{arch.c_str()};

            if (std::any_of(inputs.begin(), inputs.end(),
                            [](auto const &input) { return input.type == NVJITLINK_INPUT_LTOIR; }))
            {
                options.insert(options.end(), {"-lto", "-nvvm-latest"});
            }

            for (auto const &flag : flags)
            {
                options.push_back(flag.c_str());
            }

            CR_CHECK_NVJITLINK(api.nvJitLinkCreate(&handle, static_cast<unsigned>(options.size()), options.data()),
                               handle);
            std::vector<std::vector<char>> terminated;

            for (size_t i = 0; i < inputs.size(); ++i)
            {
                auto bytes = inputs[i].code->bytes();

                if (inputs[i].type == NVJITLINK_INPUT_PTX && bytes.back() != '\0')
                {
                    auto &ptx = terminated.emplace_back(bytes.begin(), bytes.end());
                    ptx.push_back('\0');
                    bytes = ptx;
                }

                auto name = std::format("input-{}", i);
                CR_CHECK_NVJITLINK(
                    api.nvJitLinkAddData(handle, inputs[i].type, bytes.data(), bytes.size(), name.c_str()), handle);
            }

            CR_CHECK_NVJITLINK(api.nvJitLinkComplete(handle), handle);
            size_t size{};
            CR_CHECK_NVJITLINK(api.nvJitLinkGetLinkedCubinSize(handle, &size), handle);
            std::vector<char> code(size);
            CR_CHECK_NVJITLINK(api.nvJitLinkGetLinkedCubin(handle, code.data()), handle);
            return to_blob(std::move(code));
        },
        &local);
}

/**
 * @brief One loaded CUDA library with a symbol table populated before publication.
 */
struct loaded_library
{
    CUlibrary const library;

    std::map<std::string, CUkernel, std::less<>> const kernels;

    /**
     * @brief Adopt a loaded CUDA library and its complete immutable kernel symbol table.
     * @param library CUDA library handle whose ownership is transferred to this object.
     * @param kernels All exported kernel handles keyed by their exact symbols.
     */
    loaded_library(CUlibrary library, std::map<std::string, CUkernel, std::less<>> kernels)
        : library(library), kernels(std::move(kernels))
    {
    }

    /**
     * @brief Disable copying of the uniquely owned CUDA library handle.
     */
    loaded_library(loaded_library const &) = delete;

    /**
     * @brief Disable assignment that would duplicate CUDA library ownership.
     */
    loaded_library &operator=(loaded_library const &) = delete;

    /**
     * @brief Unload the owned CUDA library after its last shared owner releases it.
     */
    ~loaded_library()
    {
        cu().cuLibraryUnload(library);
    }

    /**
     * @brief Find an already loaded kernel without locks or CUDA driver calls.
     * @param name Exact exported symbol.
     * @return Kernel handle borrowing this library.
     * @throws error If the symbol does not identify a kernel in the immutable table.
     */
    CUkernel get(std::string_view name) const
    {
        if (auto found = kernels.find(name); found != kernels.end())
        {
            return found->second;
        }

        throw error("cuda", std::format("Kernel symbol '{}' is not available in this library", name),
                    CLANGRTC_TOOLCHAIN_ERROR);
    }
};

} // namespace

#endif // CURTC_INTERNAL_TESTS

} // namespace cr

#ifndef CURTC_INTERNAL_TESTS

/**
 * @brief Owns an independent stage cache using the shared internal backend contexts.
 */
struct curtcCacheImpl
{
    cr::content_cache runtime;

    /**
     * @brief Construct an independent cache without loading CUDA or probing disk locations.
     * @param configuration Policy copied into the stage cache.
     */
    explicit curtcCacheImpl(cr::configuration configuration) : runtime(std::move(configuration))
    {
    }

    /**
     * @brief Retrieve or load a cubin and populate every exported kernel before caching the library.
     * @param code Shared finalized cubin bytes defining the loaded-library key.
     * @param local Request-local cache statistics.
     * @return Shared immutable library whose lifetime can exceed the cache lifetime.
     * @throws cr::error If loading or kernel enumeration fails; partial libraries are unloaded.
     */
    std::shared_ptr<cr::loaded_library const> load(cr::blob const &code, curtcCompileStatistics &local)
    {
        auto key = code->content_digest();
        return runtime.get_object<cr::loaded_library>(
            "cuda-library", std::span<char const>(key), code->bytes().size(),
            [&] {
                auto const &api = cr::cu();
                CUlibrary library{};
                CR_DEFER([&] {
                    if (library)
                    {
                        api.cuLibraryUnload(library);
                    }
                });
                ++local.libraryLoads;
                CR_CHECK_CUDA(
                    api.cuLibraryLoadData(&library, code->bytes().data(), nullptr, nullptr, 0, nullptr, nullptr, 0),
                    api.cuGetErrorString);
                unsigned int count{};
                CR_CHECK_CUDA(api.cuLibraryGetKernelCount(&count, library), api.cuGetErrorString);
                std::vector<CUkernel> kernels(count);

                if (count)
                {
                    CR_CHECK_CUDA(api.cuLibraryEnumerateKernels(kernels.data(), count, library), api.cuGetErrorString);
                }

                std::map<std::string, CUkernel, std::less<>> symbols;

                for (auto kernel : kernels)
                {
                    char const *name{};
                    CR_CHECK_CUDA(api.cuKernelGetName(&name, kernel), api.cuGetErrorString);
                    symbols.emplace(name, kernel);
                }

                auto result = std::make_shared<cr::loaded_library>(library, std::move(symbols));
                library = nullptr;
                return result;
            },
            &local);
    }
};

/**
 * @brief Owns per-request kernel metadata and a shared reference to an immutable CUDA library.
 */
struct curtcLibraryImpl
{
    /**
     * @brief Per-request identities are owned here, separately from shared code and loaded CUDA libraries.
     */
    struct kernel_info
    {
        std::string id;

        std::string expression;

        std::string cudaName;

        std::string symbolName;
    };

    std::atomic<size_t> references{1};

    std::shared_ptr<cr::loaded_library const> owner;

    std::vector<kernel_info> metadata;

    /**
     * @brief Bind owned request metadata to its loaded CUDA library with one initial reference.
     * @param owner Shared library retaining every borrowed CUDA kernel handle.
     * @param metadata Ordered kernel identities and names copied from the request and compiler.
     */
    curtcLibraryImpl(std::shared_ptr<cr::loaded_library const> owner, std::vector<kernel_info> metadata)
        : owner(std::move(owner)), metadata(std::move(metadata))
    {
    }
};

extern "C"
{

    clangrtcStatus curtcInit(clangrtcDiagnosticCallback callback, void *userData)
    {
        auto status = clangrtcInit(callback, userData);

        if (status != CLANGRTC_SUCCESS)
        {
            return status;
        }

        return cr::call(callback, userData, [] {
            (void)cr::nvJitLink();
            (void)cr::cu();
        });
    }

    clangrtcStatus curtcDefaultConfiguration(curtcConfiguration *configuration)
    {
        if (!configuration)
        {
            return CLANGRTC_INVALID_ARGUMENT;
        }

        *configuration = {{}, cr::default_memory_bytes, cr::default_disk_bytes, true, true, nullptr, nullptr};
        return CLANGRTC_SUCCESS;
    }

    clangrtcStatus curtcCreateCache(curtcConfiguration const *configuration, curtcCache *output)
    {
        if (output)
        {
            *output = nullptr;
        }

        return cr::call(nullptr, nullptr, [&] {
            CR_EXPECTS(output, "NULL cache output", std::invalid_argument);
            cr::configuration options;

            if (configuration)
            {
                if (configuration->cacheDirectory.size)
                {
                    options.cacheDirectory = cr::to_name(configuration->cacheDirectory);
                }

                options.memoryBytes = configuration->memoryBytes;
                options.diskBytes = configuration->diskBytes;
                options.memoryCache = configuration->memoryCache;
                options.diskCache = configuration->diskCache;

                if (configuration->diagnostic)
                {
                    auto callback = configuration->diagnostic;
                    auto user = configuration->userData;
                    options.diagnostic = [callback, user](std::string_view message) {
                        cr::diagnostic(callback, user, CLANGRTC_DIAGNOSTIC_WARNING, "cache", message);
                    };
                }
            }

            *output = new curtcCacheImpl(std::move(options));
        });
    }

    void curtcDestroyCache(curtcCache cache)
    {
        delete cache;
    }

    clangrtcStatus curtcGet(curtcCache cache, curtcRequest const *request, clangrtcDiagnosticCallback callback,
                            void *user, curtcLibrary *output, curtcCompileStatistics *statistics)
    {
        auto start = std::chrono::steady_clock::now();

        if (output)
        {
            *output = nullptr;
        }

        if (statistics)
        {
            *statistics = {};
        }

        curtcCompileStatistics local{};
        auto status = cr::call(callback, user, [&] {
            CR_EXPECTS(cache, "NULL cache", std::invalid_argument);
            CR_EXPECTS(request, "NULL request", std::invalid_argument);
            CR_EXPECTS(output, "NULL library output", std::invalid_argument);
            auto requested_inputs = std::span(request->inputs, request->inputCount);
            CR_EXPECTS(!requested_inputs.empty(), "At least one input is required", std::invalid_argument);
            auto target = cr::to_name(request->target);
            auto options = cr::to_strings(request->linkOptions, request->linkOptionCount);
            std::vector<cr::link_input> inputs;
            std::vector<curtcLibraryImpl::kernel_info> metadata;
            bool direct = requested_inputs.size() == 1 && requested_inputs.front().type == CURTC_INPUT_TYPE_CUBIN &&
                          options.empty();

            for (auto const &input : requested_inputs)
            {
                auto selections = std::span(input.entries, input.entryCount);
                std::vector<std::string> entries;

                for (auto const &entry : selections)
                {
                    auto selector = cr::to_name(entry.name);
                    CR_EXPECTS(!selector.empty(), "Kernel selectors must be nonempty", std::invalid_argument);
                    entries.push_back(std::move(selector));
                }

                auto result = cr::prepare_device_code(cache->runtime, input, entries, target, local);
                inputs.push_back({cr::to_nvJitLink_input_type(input.type), std::move(result.code)});
                bool source_kernel = input.type == CURTC_INPUT_TYPE_CUDA || input.type == CURTC_INPUT_TYPE_ARTIFACT;

                for (size_t i = 0; i < selections.size(); ++i)
                {
                    metadata.push_back({cr::to_string(selections[i].id), source_kernel ? entries[i] : std::string{},
                                        std::move(result.cuda_names[i]), std::move(result.symbols[i])});
                }
            }

            auto cubin = direct ? inputs.front().code : cr::link(cache->runtime, inputs, target, options, local);
            auto library = cache->load(cubin, local);

            for (auto const &kernel : metadata)
            {
                // Validate every input's selections against the complete, immutable export table.
                library->get(kernel.symbolName);
            }

            // The shared library and its kernel symbol table are complete before publishing the result.
            *output = new curtcLibraryImpl(std::move(library), std::move(metadata));
        });

        if (statistics)
        {
            local.totalNs = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
            *statistics = local;
        }

        return status;
    }

    clangrtcStatus curtcEnumerateKernels(curtcLibrary library, size_t *count, curtcKernel *kernels)
    {
        if (!count)
        {
            return CLANGRTC_INVALID_ARGUMENT;
        }

        if (!library)
        {
            *count = 0;
            return CLANGRTC_INVALID_ARGUMENT;
        }

        auto required = library->metadata.size();
        // A count-only query never reads the caller's count. Check capacity before writing any entries.
        auto capacity = kernels ? *count : 0;
        *count = required;

        if (!kernels)
        {
            return CLANGRTC_SUCCESS;
        }

        if (capacity < required)
        {
            return CLANGRTC_INVALID_ARGUMENT;
        }

        for (size_t i = 0; i < required; ++i)
        {
            auto const &m = library->metadata[i];
            kernels[i] = {{m.id.data(), m.id.size()},
                          {m.expression.data(), m.expression.size()},
                          {m.cudaName.data(), m.cudaName.size()},
                          {m.symbolName.data(), m.symbolName.size()}};
        }

        return CLANGRTC_SUCCESS;
    }

    clangrtcStatus curtcGetKernel(curtcLibrary library, clangrtcString symbol, CUkernel *output)
    {
        if (output)
        {
            *output = nullptr;
        }

        return cr::call(nullptr, nullptr, [&] {
            CR_EXPECTS(library, "NULL library", std::invalid_argument);
            CR_EXPECTS(output, "NULL kernel output", std::invalid_argument);
            auto selector = cr::to_name(symbol);
            CR_EXPECTS(!selector.empty(), "Kernel symbol must be nonempty", std::invalid_argument);
            *output = library->owner->get(selector);
        });
    }

    void curtcRetainLibrary(curtcLibrary library)
    {
        if (library)
        {
            library->references.fetch_add(1, std::memory_order_relaxed);
        }
    }

    void curtcReleaseLibrary(curtcLibrary library)
    {
        if (library && library->references.fetch_sub(1, std::memory_order_acq_rel) == 1)
        {
            delete library;
        }
    }

    clangrtcStatus curtcGetStatistics(curtcCache cache, curtcCacheStatistics *statistics)
    {
        if (statistics)
        {
            *statistics = {};
        }

        if (!cache || !statistics)
        {
            return CLANGRTC_INVALID_ARGUMENT;
        }

        auto s = cache->runtime.stats.get();
        *statistics = {s.memoryHits,           s.diskHits,   s.misses,
                       s.sharedRequests,       s.diskErrors, s.kernelCompilations,
                       s.fragmentCompilations, s.links};
        return CLANGRTC_SUCCESS;
    }

} // extern "C"

#endif // CURTC_INTERNAL_TESTS
