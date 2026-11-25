/* SPDX-License-Identifier: Apache-2.0 */

/**
 * @file curtc_test.cpp
 * @brief Test internal runtime cache primitives, binary storage, and callbacks.
 *
 * Exercises memory and disk reuse, concurrent work sharing, corruption recovery, and eviction without GPU execution.
 */

#include "curtc.cpp"
#include "support.hpp"
#include "test_config.hpp"

#include <array>
#include <future>
#include <latch>

namespace
{
cr::configuration local(test::TemporaryDirectory const &dir)
{
    cr::configuration c;
    c.cacheDirectory = dir.path;
    return c;
}

/**
 * @brief Borrow noncopyable cache factories and invoke plain function-pointer callbacks.
 *
 * Uses a lambda owning a unique pointer and a separate function pointer to check the function_ref-backed cache API.
 */
TEST(Curtcallbacks, BorrowsNoncopyableFactoryAndInvokesFunctionPointers)
{
    test::TemporaryDirectory dir;
    auto config = local(dir);
    config.diskCache = false;
    cr::content_cache cache(config);
    auto factory = [owned = std::make_unique<int>(42)] { return cr::to_blob(std::format("{}", *owned)); };
    auto result = cache.get("callback", "noncopyable", factory);
    EXPECT_TRUE(std::ranges::equal(result->bytes(), std::string_view("42")));
    auto function = +[] { return cr::to_blob(std::string_view("function")); };
    EXPECT_TRUE(std::ranges::equal(cache.get("callback", "pointer", function)->bytes(), std::string_view("function")));
}

/**
 * @brief Keep decimal and binary byte literals distinct through terabyte units.
 *
 * Checks exact unit scales and the default disk-cache budget without relying on floating-point conversion.
 */
TEST(CurtcBytes, DecimalAndBinaryUnitsRemainDistinct)
{
    using namespace cr;
    constexpr std::array decimal{1_B, 1_KB, 1_MB, 1_GB, 1_TB};

    constexpr std::array expected_decimal{1ULL, 1000ULL, 1'000'000ULL, 1'000'000'000ULL, 1'000'000'000'000ULL};

    EXPECT_TRUE(std::ranges::equal(decimal, expected_decimal));
    constexpr std::array binary{1_KiB, 1_MiB, 1_GiB, 1_TiB};

    constexpr std::array expected_binary{1ULL << 10, 1ULL << 20, 1ULL << 30, 1ULL << 40};

    EXPECT_TRUE(std::ranges::equal(binary, expected_binary));
    EXPECT_NE(1_MB, 1_MiB);
    EXPECT_EQ(default_disk_bytes, 4_GiB);
}

/**
 * @brief Own exact binary ranges independently of source-buffer mutation or transfer.
 *
 * Covers borrowed subspans, moved buffers, embedded NUL bytes, byte views, and empty binary values.
 */
TEST(CurtcBinary, OwnsExactByteRanges)
{
    std::vector<char> bytes{'x', '\0', static_cast<char>(0xff), 'y', '\0'};
    auto copied = cr::to_blob(std::span<char const>(bytes).subspan(1, 3));
    bytes[1] = 'z';
    std::vector<char> expected{'\0', static_cast<char>(0xff), 'y'};
    EXPECT_TRUE(std::ranges::equal(copied->bytes(), expected));
    EXPECT_TRUE(std::ranges::equal(copied->to_view(), expected));
    auto moved = cr::to_blob(std::move(expected));
    EXPECT_TRUE(std::ranges::equal(moved->bytes(), copied->bytes()));
    EXPECT_TRUE(cr::to_blob(std::span<char const>{})->bytes().empty());
}

/**
 * @brief Round-trip binary sections through serialization and persistent caching.
 *
 * Includes empty sections and embedded NUL bytes, then checks a disk hit without regenerating the payload.
 */
TEST(Curtcache, PreservesBinarySectionsAndEmptyBuffers)
{
    std::vector<char> expected{'\0', static_cast<char>(0xff), '\0', 'x'};
    cr::writer w;
    w.text("metadata");
    w.bytes(expected);
    w.bytes({});
    auto package = cr::envelope("BINARY\n", w.data);
    cr::reader r{cr::unpack(package, "BINARY\n")};
    EXPECT_EQ(r.text(), "metadata");
    EXPECT_TRUE(std::ranges::equal(r.bytes(), expected));
    EXPECT_TRUE(r.bytes().empty());
    r.end();

    test::TemporaryDirectory dir;
    auto cfg = local(dir);
    cfg.memoryCache = false;
    cr::content_cache cache(cfg);
    int calls{};
    auto generate = [&] {
        ++calls;
        return cr::to_blob(expected);
    };

    auto first = cache.get("binary", expected, generate);
    auto restored = cache.get("binary", expected, generate);
    EXPECT_EQ(calls, 1);
    EXPECT_EQ(cache.stats.diskHits, 1);
    EXPECT_TRUE(std::ranges::equal(restored->bytes(), first->bytes()));
    EXPECT_TRUE(std::ranges::equal(restored->bytes(), expected));
}

/**
 * @brief Share identical in-flight cache work without caching failures.
 *
 * Starts simultaneous requests, verifies one shared result, and retries failed factories before a successful recovery.
 */
TEST(Curtcache, DeduplicatesConcurrentWorkAndRetriesFailures)
{
    test::TemporaryDirectory dir;
    auto cfg = local(dir);
    cfg.diskCache = false;
    cr::content_cache cache(cfg);
    std::latch start(8);
    std::atomic<int> calls{};
    std::vector<std::future<cr::blob>> jobs;

    for (int i = 0; i < 8; ++i)
    {
        jobs.push_back(std::async(std::launch::async, [&] {
            start.count_down();
            start.wait();
            return cache.get("stage", "key", [&] {
                ++calls;
                return cr::to_blob(std::string_view("payload"));
            });
        }));
    }

    auto first = jobs.front().get();

    for (size_t i = 1; i < jobs.size(); ++i)
    {
        EXPECT_EQ(first, jobs[i].get());
    }

    EXPECT_EQ(calls, 1);

    for (int i = 0; i < 2; ++i)
    {
        EXPECT_THROW(cache.get("bad", "key", []() -> cr::blob { throw std::runtime_error("retry"); }),
                     std::runtime_error);
    }

    auto recovered = cache.get("bad", "key", [] { return cr::to_blob(std::string_view("recovered")); });
    EXPECT_TRUE(std::ranges::equal(recovered->bytes(), std::string_view("recovered")));
}

/**
 * @brief Recover corrupt disk entries and evict cached values without invalidating caller ownership.
 *
 * Exercises disk-hit reuse, regeneration after corruption, memory eviction, and enforcement of the persistent-byte
 * budget.
 */
TEST(Curtcache, DiskCorruptionFallbackAndEvictionPreserveResults)
{
    test::TemporaryDirectory dir;
    auto cfg = local(dir);
    cfg.memoryCache = false;
    cr::content_cache cache(cfg);
    int calls{};
    auto generate = [&] {
        ++calls;
        return cr::to_blob(std::string_view("payload"));
    };

    cache.get("stage", "key", generate);
    EXPECT_TRUE(std::ranges::equal(cache.get("stage", "key", generate)->bytes(), std::string_view("payload")));
    EXPECT_EQ(calls, 1);

    for (auto const &f : std::filesystem::directory_iterator(dir.path / "v1"))
    {
        std::ofstream out(f.path());
        out << "corrupt";
    }

    EXPECT_TRUE(std::ranges::equal(cache.get("stage", "key", generate)->bytes(), std::string_view("payload")));
    EXPECT_EQ(calls, 2);
    EXPECT_GT(cache.stats.diskErrors, 0);
    cfg.cacheDirectory = dir.path / "memory";
    cfg.diskCache = false;
    cfg.memoryCache = true;
    cfg.memoryBytes = 7;
    cr::content_cache memory(cfg);
    auto retained = memory.get("stage", "one", generate);
    memory.get("stage", "two", generate);
    EXPECT_TRUE(std::ranges::equal(retained->bytes(), std::string_view("payload")));
    auto count = calls;
    memory.get("stage", "one", generate);
    EXPECT_GT(calls, count);
    cfg.diskCache = true;
    cfg.cacheDirectory = dir.path / "bounded";
    cfg.diskBytes = 300;
    cfg.memoryCache = false;
    cr::content_cache bounded(cfg);

    for (int i = 0; i < 8; ++i)
    {
        bounded.get("bounded", std::to_string(i), generate);
    }

    size_t size{};

    for (auto const &f : std::filesystem::directory_iterator(*cfg.cacheDirectory / "v1"))
    {
        size += f.file_size();
    }

    EXPECT_LE(size, cfg.diskBytes);
}

} // namespace
