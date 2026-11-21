/* SPDX-License-Identifier: Apache-2.0 */

/**
 * @file prepared_diagnostics_test.cpp
 * @brief Preserve CUDA diagnostics and finish pending template work when preparing compiler artifacts.
 */

#include "support.hpp"

#include <format>

namespace test
{
/** @brief Exercise CUDA semantic checks across PCH, full BMI, and reduced BMI boundaries. */
class PreparedDiagnosticsTest : public CompilationTest, public ::testing::WithParamInterface<std::string>
{
  protected:
    /** @brief Prepare declarations and expose the tested function from named-module artifacts. */
    Operation prepare_declarations(std::string_view declarations)
    {
        auto module = GetParam() != "pch";
        auto source = module ? std::string("module;\n#include <clangrtc/device.hpp>\n") : std::string{};
        source.append(declarations);

        if (module)
        {
            source.append("\nexport module checked_environment;\n"
                          "export namespace checked { using ::checked::invoke; }\n");
        }

        auto flags = options;
        flags.push_back(std::format("--artifact={}", GetParam()));
        auto input = request(source, {});
        input.source.path = to_view("/__clangrtc__/checked_environment.cppm");
        return prepare(input, flags);
    }

    /** @brief Compile a kernel invoking the prepared function with a caller-selected argument. */
    Operation consume(clangrtcArtifact artifact, std::string_view argument, std::string_view declarations)
    {
        auto source = GetParam() == "pch" ? std::string{} : std::string("import checked_environment;\n");
        source.append(declarations);
        source.append(std::format("extern \"C\" __global__ void checked_kernel(int *out) {{ "
                                  "*out = checked::invoke({}); }}\n",
                                  argument));
        auto slot = to_view("&checked_kernel");
        auto input = request(source, {&slot, 1});
        input.artifact = artifact;
        return compile(input, {});
    }
};

/** @brief Accept a valid imported specialization and reject a newly instantiated device-to-host call. */
TEST_P(PreparedDiagnosticsTest, ChecksNewTemplateSpecializations)
{
    auto prepared = prepare_declarations(R"CUDA(
        namespace checked
        {
        __host__ int host_only(int);

        template <class T> __device__ int invoke(T value)
        {
            if constexpr (sizeof(T) == sizeof(int))
            {
                return value + 1;
            }
            else
            {
                return host_only(value);
            }
        }
        } // namespace checked
    )CUDA");
    ASSERT_EQ(prepared.status, CLANGRTC_SUCCESS) << prepared.log();
    auto loaded = load(prepared.output());
    ASSERT_EQ(loaded.status, CLANGRTC_SUCCESS);
    auto valid = consume(loaded.artifact.get(), "7", {});
    ASSERT_EQ(valid.status, CLANGRTC_SUCCESS) << valid.log();
    EXPECT_FALSE(valid.output().empty());
    auto invalid = consume(loaded.artifact.get(), "7.0", {});
    EXPECT_EQ(invalid.status, CLANGRTC_COMPILATION_FAILED) << invalid.log();
    EXPECT_NE(invalid.log().find("host_only"), std::string::npos);
    EXPECT_TRUE(invalid.output().empty());
}

/** @brief Repeated requests can independently define the same names and change preprocessing state. */
TEST_P(PreparedDiagnosticsTest, RequestDefinitionsAndMacrosStayIndependent)
{
    auto prepared = prepare_declarations("namespace checked { __device__ int invoke(int x) { return x; } }\n");
    ASSERT_EQ(prepared.status, CLANGRTC_SUCCESS) << prepared.log();
    auto loaded = load(prepared.output());
    ASSERT_EQ(loaded.status, CLANGRTC_SUCCESS);

    for (int value : {3, 7})
    {
        auto declarations = std::format("#ifdef REQUEST_VALUE\n#error Request macro leaked\n#endif\n"
                                        "#define REQUEST_VALUE {}\n"
                                        "struct Local {{ int value = REQUEST_VALUE; }};\n"
                                        "__device__ int request_value() {{ return Local{{}}.value; }}\n",
                                        value);
        auto result = consume(loaded.artifact.get(), "request_value()", declarations);
        ASSERT_EQ(result.status, CLANGRTC_SUCCESS) << result.log();
        EXPECT_GT(result.info().frontendNs, 0);
        EXPECT_LE(result.info().frontendNs, result.info().totalElapsedNs);
        EXPECT_FALSE(result.output().empty());
    }

    auto result = consume(loaded.artifact.get(), "17", "#ifdef REQUEST_VALUE\n#error Request macro leaked\n#endif\n");
    EXPECT_EQ(result.status, CLANGRTC_SUCCESS) << result.log();
}

/** @brief Failed consumers release their macros, diagnostics, and incomplete declarations before the next request. */
TEST_P(PreparedDiagnosticsTest, FailedRequestDoesNotAffectLaterCompilation)
{
    auto prepared = prepare_declarations("namespace checked { __device__ int invoke(int x) { return x; } }\n");
    ASSERT_EQ(prepared.status, CLANGRTC_SUCCESS) << prepared.log();
    auto loaded = load(prepared.output());
    ASSERT_EQ(loaded.status, CLANGRTC_SUCCESS);
    auto invalid = consume(loaded.artifact.get(), "missing()", "#define FAILED_REQUEST 1\nstruct Local;\n");
    EXPECT_EQ(invalid.status, CLANGRTC_COMPILATION_FAILED) << invalid.log();
    EXPECT_GT(invalid.info().frontendNs, 0);
    EXPECT_TRUE(invalid.output().empty());
    auto recovered = consume(loaded.artifact.get(), "request_value()",
                             "#ifdef FAILED_REQUEST\n#error Failed request macro leaked\n#endif\n"
                             "struct Local { int value = 29; };\n"
                             "__device__ int request_value() { return Local{}.value; }\n");
    EXPECT_EQ(recovered.status, CLANGRTC_SUCCESS) << recovered.log();
    EXPECT_FALSE(recovered.output().empty());
}

/** @brief Keep unused prepared host/device code valid and diagnose new deferred errors in importing consumers. */
TEST_P(PreparedDiagnosticsTest, ChecksDeferredErrorsInConsumers)
{
    auto prepared = prepare_declarations(R"CUDA(
        namespace checked
        {
        __host__ int host_only(int);

        __host__ __device__ inline int unused_host_device(int value)
        {
            return host_only(value);
        }

        __device__ int invoke(int value)
        {
            return value + 1;
        }
        } // namespace checked
    )CUDA");
    ASSERT_EQ(prepared.status, CLANGRTC_SUCCESS) << prepared.log();
    auto loaded = load(prepared.output());
    ASSERT_EQ(loaded.status, CLANGRTC_SUCCESS);
    auto valid = consume(loaded.artifact.get(), "7", {});
    ASSERT_EQ(valid.status, CLANGRTC_SUCCESS) << valid.log();
    auto invalid = consume(loaded.artifact.get(), "consumer_helper(7)", R"CUDA(
        __host__ int consumer_host_only(int);

        __host__ __device__ inline int consumer_helper(int value)
        {
            return consumer_host_only(value);
        }
    )CUDA");
    EXPECT_EQ(invalid.status, CLANGRTC_COMPILATION_FAILED) << invalid.log();
    EXPECT_NE(invalid.log().find("consumer_host_only"), std::string::npos);
    EXPECT_TRUE(invalid.output().empty());
}

/** @brief Reject a statically invalid device-only definition before producing an artifact. */
TEST_P(PreparedDiagnosticsTest, RejectsInvalidDefinitionsDuringPreparation)
{
    auto prepared = prepare_declarations(R"CUDA(
        namespace checked
        {
        __host__ int host_only(int);

        __device__ int invoke(int value)
        {
            return host_only(value);
        }
        } // namespace checked
    )CUDA");
    EXPECT_EQ(prepared.status, CLANGRTC_COMPILATION_FAILED) << prepared.log();
    EXPECT_NE(prepared.log().find("host_only"), std::string::npos);
    EXPECT_TRUE(prepared.output().empty());
}

/** @brief Cover full and reduced module serialization together with ordinary PCH preparation. */
INSTANTIATE_TEST_SUITE_P(Formats, PreparedDiagnosticsTest, ::testing::Values("pch", "bmi", "reduced-bmi"));

/** @brief Preserve exported template definitions when a clean reduced BMI can prune diagnostic-only roots. */
TEST_F(CompilationTest, ReducedBmiRetainsTemplatesInCleanCudaEnvironment)
{
    auto flags = options;
    flags.push_back("--clang-option=-nocudainc");
    flags.push_back("--artifact=reduced-bmi");
    auto input = request(R"CUDA(
                             module;

                             namespace implementation
                             {
                             template <class T> __attribute__((device)) T adjust(T value)
                             {
                                 return value + T{1};
                             }

                             __attribute__((device)) int unrelated(int value)
                             {
                                 return value + 10;
                             }
                             } // namespace implementation

                             export module clean_environment;

                             export namespace checked
                             {
                             using ::implementation::adjust;
                             }
                         )CUDA",
                         {});
    input.source.path = to_view("/__clangrtc__/clean_environment.cppm");
    auto prepared = prepare(input, flags);
    ASSERT_EQ(prepared.status, CLANGRTC_SUCCESS) << prepared.log();
    auto loaded = load(prepared.output());
    ASSERT_EQ(loaded.status, CLANGRTC_SUCCESS);
    auto slot = to_view("&pruned_consumer");
    auto consumer = request(R"CUDA(
import clean_environment;

                                extern "C" __attribute__((global)) void pruned_consumer(double *out)
                                {
                                    *out = checked::adjust(41.0);
                                }
                            )CUDA",
                            {&slot, 1});
    consumer.artifact = loaded.artifact.get();
    auto result = compile(consumer, {});
    ASSERT_EQ(result.status, CLANGRTC_SUCCESS) << result.log();
    EXPECT_FALSE(result.output().empty());
    ASSERT_EQ(result.info().symbolCount, 1);
    EXPECT_EQ(to_string(result.info().symbols[0].name), "pruned_consumer");
}

/** @brief Diagnose pending template errors during PCH preparation rather than deferring them to a consumer. */
TEST(PchPreparation, InstantiatesPendingTemplatesAheadOfTime)
{
    auto result = prepare(request(R"CPP(
        template <class T> int pending()
        {
            static_assert(sizeof(T) == 0, "AOT pending template instantiation");
            return 0;
        }

        inline int use_pending()
        {
            return pending<int>();
        }
    )CPP",
                                  {}),
                          {"--host-only", "--artifact=pch"});
    EXPECT_EQ(result.status, CLANGRTC_COMPILATION_FAILED) << result.log();
    EXPECT_NE(result.log().find("AOT pending template instantiation"), std::string::npos);
    EXPECT_TRUE(result.output().empty());
}

} // namespace test
