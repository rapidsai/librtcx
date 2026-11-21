/* SPDX-License-Identifier: Apache-2.0 */

/**
 * @file curtc_cache_directory_test.cpp
 * @brief Test persistent-cache directory selection with controlled environment variables.
 *
 * Builds isolated filesystem candidates to check precedence, access failures, versioned subdirectories, and disabled
 * disk caching.
 */

#include "curtc.cpp"
#include "support.hpp"

#include <format>
#include <optional>

namespace
{
class CurtcCacheDirectory : public ::testing::Test
{
  protected:
    test::TemporaryDirectory temporary;

    std::map<std::string, std::string> environment;

    std::vector<std::string> warnings;

    std::vector<std::filesystem::path> restricted;

    void SetUp() override
    {
        environment = {{"XDG_CACHE_HOME", (temporary.path / "xdg").string()},
                       {"HOME", (temporary.path / "home").string()},
                       {"TMPDIR", (temporary.path / "temporary").string()}};
        std::filesystem::create_directories(environment.at("HOME"));
        std::filesystem::create_directories(environment.at("TMPDIR"));
    }

    void TearDown() override
    {
        for (auto const &path : restricted)
        {
            std::error_code ec;
            std::filesystem::permissions(path, std::filesystem::perms::owner_all, ec);
        }
    }

    auto environment_getter()
    {
        return [this](std::string const &name) -> std::optional<std::string> {
            auto found = environment.find(name);
            return found == environment.end() ? std::nullopt : std::optional<std::string>(found->second);
        };
    }

    std::optional<std::filesystem::path> select(std::optional<std::filesystem::path> preferred)
    {
        auto getter = environment_getter();
        auto candidates = cr::cache_directory_candidates(getter);
        return cr::select_cache_directory(preferred, candidates, getter,
                                          [&](std::string_view message) { warnings.emplace_back(message); });
    }

    void block(std::filesystem::path const &path)
    {
        std::filesystem::create_directories(path.parent_path());
        std::ofstream file(path);
        file << "not a directory";
        ASSERT_TRUE(file);
    }

    std::filesystem::path user_temporary(std::filesystem::path const &root)
    {
        return root / std::format("curtc-{}", ::getuid()) / "v1";
    }
};

/**
 * @brief Prefer an explicit cache root over environment-based candidates.
 *
 * Checks the versioned subdirectory, absence of leftover probe files, and that lower-priority directories are not
 * created.
 */
TEST_F(CurtcCacheDirectory, PrefersConfiguredRootAndLeavesNoProbeFiles)
{
    auto preferred = temporary.path / "configured";
    environment["CURTC_CACHE_DIR"] = (temporary.path / "override").string();
    auto selected = select(preferred);
    EXPECT_EQ(selected, preferred / "v1");
    ASSERT_TRUE(selected);
    EXPECT_TRUE(std::filesystem::is_empty(*selected));
    EXPECT_FALSE(std::filesystem::exists(environment.at("CURTC_CACHE_DIR")));
    EXPECT_FALSE(std::filesystem::exists(environment.at("XDG_CACHE_HOME")));
    EXPECT_TRUE(warnings.empty());
}

/**
 * @brief Use CURTC_CACHE_DIR when no explicit cache root is supplied.
 *
 * Checks that the environment override receives the versioned subdirectory before standard fallback locations are
 * considered.
 */
TEST_F(CurtcCacheDirectory, UsesEnvironmentOverrideWhenNoRootIsConfigured)
{
    environment["CURTC_CACHE_DIR"] = (temporary.path / "override").string();
    EXPECT_EQ(select({}), std::filesystem::path(environment.at("CURTC_CACHE_DIR")) / "v1");
    EXPECT_FALSE(std::filesystem::exists(environment.at("XDG_CACHE_HOME")));
}

/**
 * @brief Distinguish absent environment values from present empty paths.
 *
 * Exercises the optional-valued environment-path adapter so emptiness is not used as a substitute for an unset
 * variable.
 */
TEST_F(CurtcCacheDirectory, DistinguishesUnsetAndEmptyEnvironmentPaths)
{
    auto getter = environment_getter();
    EXPECT_FALSE(cr::get_envpath("CURTC_CACHE_DIR", getter));
    environment["XDG_CACHE_HOME"] = "";
    auto root = cr::get_envpath("XDG_CACHE_HOME", getter);
    ASSERT_TRUE(root);
    EXPECT_EQ(*root, std::filesystem::path{});
    auto candidates = cr::cache_directory_candidates(getter);
    ASSERT_FALSE(candidates.empty());
    EXPECT_EQ(candidates.front(), std::filesystem::path("curtc"));
    environment.erase("XDG_CACHE_HOME");
    EXPECT_FALSE(cr::get_envpath("XDG_CACHE_HOME", getter));
}

/**
 * @brief Fall back to XDG_CACHE_HOME when the configured root is absent or unusable.
 *
 * Checks the curtc versioned directory and warning behavior for each preferred-root case.
 */
TEST_F(CurtcCacheDirectory, UsesXdgWhenRootIsAbsentOrUnusable)
{
    auto expected = std::filesystem::path(environment.at("XDG_CACHE_HOME")) / "curtc/v1";
    EXPECT_EQ(select({}), expected);
    auto blocked = temporary.path / "blocked";
    block(blocked);
    EXPECT_EQ(select(blocked), expected);
    EXPECT_EQ(warnings.size(), 1);
    EXPECT_FALSE(std::filesystem::exists(std::filesystem::path(environment.at("HOME")) / ".cache"));
}

/**
 * @brief Use the home cache directory when the XDG cache location cannot be used.
 *
 * Blocks the higher-priority path and checks both the selected fallback and its warning count.
 */
TEST_F(CurtcCacheDirectory, FallsBackFromUnusableXdgToHome)
{
    block(environment.at("XDG_CACHE_HOME"));
    auto expected = std::filesystem::path(environment.at("HOME")) / ".cache/curtc/v1";
    EXPECT_EQ(select({}), expected);
    EXPECT_EQ(warnings.size(), 1);
}

/**
 * @brief Use TMPDIR after both XDG and home cache locations fail.
 *
 * Blocks the higher-priority cache subdirectories and verifies the user-specific temporary-cache path and warnings.
 */
TEST_F(CurtcCacheDirectory, FallsBackFromUnusableHomeCacheToTmpdir)
{
    block(environment.at("XDG_CACHE_HOME"));
    block(std::filesystem::path(environment.at("HOME")) / ".cache");
    EXPECT_EQ(select({}), user_temporary(environment.at("TMPDIR")));
    EXPECT_EQ(warnings.size(), 2);
}

/**
 * @brief Skip nonexistent HOME and TMPDIR roots without creating them.
 *
 * Requires an accessible system temporary directory and checks that missing environment roots do not produce spurious
 * warnings.
 */
TEST_F(CurtcCacheDirectory, SkipsMissingHomeAndTmpdirAndUsesSystemTmp)
{
    if (::access("/tmp", R_OK | W_OK | X_OK) != 0)
    {
        GTEST_SKIP() << "Requires an accessible /tmp";
    }

    environment.erase("XDG_CACHE_HOME");
    environment["HOME"] = (temporary.path / "missing-home").string();
    environment["TMPDIR"] = (temporary.path / "missing-temporary").string();
    EXPECT_EQ(select({}), user_temporary("/tmp"));
    EXPECT_FALSE(std::filesystem::exists(environment.at("HOME")));
    EXPECT_FALSE(std::filesystem::exists(environment.at("TMPDIR")));
    EXPECT_TRUE(warnings.empty());
}

/**
 * @brief Use the system temporary directory when the TMPDIR cache subfolder is blocked.
 *
 * Skips unavailable XDG/home roots and checks the user-specific /tmp fallback and warning count.
 */
TEST_F(CurtcCacheDirectory, FallsBackFromUnusableTmpdirSubfolderToSystemTmp)
{
    if (::access("/tmp", R_OK | W_OK | X_OK) != 0)
    {
        GTEST_SKIP() << "Requires an accessible /tmp";
    }

    environment.erase("XDG_CACHE_HOME");
    environment.erase("HOME");
    auto blocked = user_temporary(environment.at("TMPDIR")).parent_path();
    block(blocked);
    EXPECT_EQ(select({}), user_temporary("/tmp"));
    EXPECT_EQ(warnings.size(), 1);
}

/**
 * @brief Exclude HOME and TMPDIR values that name ordinary files.
 *
 * Builds candidate roots with controlled filesystem entries and checks whether the accessible system temporary
 * directory remains.
 */
TEST_F(CurtcCacheDirectory, SkipsNonDirectoryHomeAndTmpdir)
{
    environment.erase("XDG_CACHE_HOME");
    environment["HOME"] = (temporary.path / "home-file").string();
    environment["TMPDIR"] = (temporary.path / "temporary-file").string();
    block(environment.at("HOME"));
    block(environment.at("TMPDIR"));
    auto candidates = cr::cache_directory_candidates(environment_getter());
    EXPECT_EQ(candidates.size(), ::access("/tmp", R_OK | X_OK) == 0 ? 1 : 0);

    if (!candidates.empty())
    {
        EXPECT_EQ(candidates.front(), user_temporary("/tmp").parent_path());
    }
}

/**
 * @brief Reject a cache directory that lacks write access for the current user.
 *
 * Removes directory write permission and checks XDG fallback, skipping when the user can bypass the permission
 * restriction.
 */
TEST_F(CurtcCacheDirectory, SkipsUnwritableDirectory)
{
    auto denied = temporary.path / "denied";
    std::filesystem::create_directories(denied);
    restricted.push_back(denied);
    std::filesystem::permissions(denied, std::filesystem::perms::owner_read | std::filesystem::perms::owner_exec);

    if (::access(denied.c_str(), W_OK) == 0)
    {
        GTEST_SKIP() << "Current user can bypass directory write permissions";
    }

    EXPECT_EQ(select(denied), std::filesystem::path(environment.at("XDG_CACHE_HOME")) / "curtc/v1");
    EXPECT_EQ(warnings.size(), 1);
}

/**
 * @brief Disable persistent caching when every supplied cache-root candidate is unusable.
 *
 * Supplies two blocked roots and checks an empty optional result with one warning per failure.
 */
TEST_F(CurtcCacheDirectory, ReturnsNoDirectoryWhenAllCandidatesFail)
{
    auto first = temporary.path / "first";
    auto second = temporary.path / "second";
    block(first);
    block(second);
    std::vector<std::filesystem::path> candidates{first, second};
    auto selected = cr::select_cache_directory({}, candidates, environment_getter(),
                                               [&](std::string_view message) { warnings.emplace_back(message); });
    EXPECT_FALSE(selected);
    EXPECT_EQ(warnings.size(), 2);
}

/**
 * @brief Avoid filesystem probing when disk caching is disabled or its budget is zero.
 *
 * Verifies memory reuse, no disk errors, and no creation of the configured cache directory for either policy.
 */
TEST_F(CurtcCacheDirectory, DisabledDiskCacheDoesNotProbeOrCreateDirectories)
{
    for (bool enabled : {false, true})
    {
        cr::configuration config;
        config.cacheDirectory = temporary.path / (enabled ? "zero-budget" : "disabled");
        config.diskCache = enabled;

        if (enabled)
        {
            config.diskBytes = 0;
        }

        cr::content_cache cache(config);
        int calls{};
        auto generate = [&] {
            ++calls;
            return cr::to_blob(std::string_view("value"));
        };

        auto first = cache.get("stage", "key", generate);
        EXPECT_EQ(first, cache.get("stage", "key", generate));
        EXPECT_EQ(calls, 1);
        EXPECT_EQ(cache.stats.diskErrors, 0);
        EXPECT_FALSE(std::filesystem::exists(*config.cacheDirectory));
    }
}
} // namespace
