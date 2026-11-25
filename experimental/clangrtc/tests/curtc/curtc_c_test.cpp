/* SPDX-License-Identifier: Apache-2.0 */

/**
 * @file curtc_c_test.cpp
 * @brief Test cached CUDA compilation and loaded-library ownership through the public C API.
 *
 * Uses real compiler/linker backends and the real CUDA driver and runtime to validate selections, metadata, statistics,
 * failure recovery, and shared lifetimes.
 */

#include "gpu_support.hpp"
#include "support.hpp"
#include "test_config.hpp"

#include <cstring>
#include <curtc.h>
#include <elf.h>
#include <format>
#include <fstream>
#include <future>
#include <latch>

extern "C" clangrtcStatus create_curtc_library_from_c(clangrtcString image, curtcLibrary *output);

namespace
{
namespace inputs
{
using test::to_view;
using test::to_views;

/** @brief Own source text and options borrowed by synchronous public API requests. */
struct Source
{
    /** @brief Complete CUDA translation unit. */
    std::string text;

    /** @brief Ordered compiler options. */
    test::Options options;

    /** @brief Additional definitions to retain. */
    test::Options expressions;
};

/** @brief Borrow a source; both this holder and its source must outlive the API call. */
struct source_view
{
    /** @brief Borrowed compiler-option views. */
    std::vector<clangrtcString> options;

    /** @brief Borrowed retention-expression views. */
    std::vector<clangrtcString> expressions;

    /** @brief Public input referencing the owned view arrays. */
    curtcInput value{};

    /** @brief Construct the public CUDA source input without depending on compiler internals. */
    explicit source_view(Source const &input)
        : options(to_views(input.options)), expressions(to_views(input.expressions))
    {
        value.type = CURTC_INPUT_TYPE_CUDA;
        value.source = test::request(input.text, expressions);
        value.options = options.data();
        value.optionCount = options.size();
    }
};

} // namespace inputs

using Cache = std::unique_ptr<curtcCacheImpl, decltype(&curtcDestroyCache)>;

using Library = std::unique_ptr<curtcLibraryImpl, decltype(&curtcReleaseLibrary)>;

struct diagnostic
{
    std::string stage;

    std::string message;

    static void collect(clangrtcDiagnostic const *input, void *context)
    {
        auto &out = *static_cast<diagnostic *>(context);
        out.stage = test::to_string(input->stage);
        out.message.append(test::to_string(input->message));
    }
};

Cache cache(test::TemporaryDirectory const &directory)
{
    curtcConfiguration config{};
    EXPECT_EQ(curtcDefaultConfiguration(&config), CLANGRTC_SUCCESS);
    auto path = directory.path.string();
    config.cacheDirectory = inputs::to_view(path);
    curtcCache result{};
    EXPECT_EQ(curtcCreateCache(&config, &result), CLANGRTC_SUCCESS);
    return Cache(result, curtcDestroyCache);
}

Library get_library(curtcCache cache, curtcRequest const &request, curtcCompileStatistics *stats)
{
    curtcLibrary result{};
    diagnostic log;
    auto status = curtcGet(cache, &request, diagnostic::collect, &log, &result, stats);

    if (status != CLANGRTC_SUCCESS)
    {
        throw std::runtime_error(log.message);
    }

    return Library(result, curtcReleaseLibrary);
}

Library get_library(curtcCache cache, curtcRequest const &request)
{
    return get_library(cache, request, nullptr);
}

inputs::Source source()
{
    inputs::Source input;
    input.text = "template<class T> __global__ void fixed(T* out) { *out = T{42}; }";
    input.options = {std::format("--cuda-path={}", test::CudaPath)};
    return input;
}

curtcRequest request(curtcInput &input)
{
    curtcRequest result{};
    result.inputs = &input;
    result.inputCount = 1;
    static curtcKernelSelection const source_entry{inputs::to_view("&fixed<int>"), {}};
    static curtcKernelSelection const binary_entry{inputs::to_view("fixed"), {}};
    input.entries =
        input.type == CURTC_INPUT_TYPE_CUDA || input.type == CURTC_INPUT_TYPE_ARTIFACT ? &source_entry : &binary_entry;
    input.entryCount = 1;
    result.target = inputs::to_view(test::gpu_target());
    return result;
}

curtcCacheStatistics statistics(curtcCache cache)
{
    curtcCacheStatistics result{};
    EXPECT_EQ(curtcGetStatistics(cache, &result), CLANGRTC_SUCCESS);
    return result;
}

curtcKernel kernel_info(curtcLibrary library, size_t index)
{
    size_t count{};
    EXPECT_EQ(curtcEnumerateKernels(library, &count, nullptr), CLANGRTC_SUCCESS);
    std::vector<curtcKernel> kernels(count);
    EXPECT_EQ(curtcEnumerateKernels(library, &count, kernels.data()), CLANGRTC_SUCCESS);
    return kernels.at(index);
}

CUkernel kernel_handle(curtcLibrary library, size_t index)
{
    CUkernel result{};
    EXPECT_EQ(curtcGetKernel(library, kernel_info(library, index).symbolName, &result), CLANGRTC_SUCCESS);
    return result;
}

/**
 * @brief Keep a C-owned loaded kernel alive after its cache is destroyed.
 *
 * Creates the library in a C translation unit, then uses GoogleTest assertions for
 * status, metadata enumeration, symbol lookup, reference counting, and execution after cache destruction.
 */
TEST(Curtc, CTranslationUnitOwnsLoadedKernelAfterCacheDestruction)
{
    curtcLibrary raw{};
    auto image = test::cubin_fixture();
    auto status = create_curtc_library_from_c({image.data(), image.size()}, &raw);
    Library library(raw, curtcReleaseLibrary);
    ASSERT_EQ(status, CLANGRTC_SUCCESS);
    ASSERT_NE(library.get(), nullptr);
    curtcRetainLibrary(library.get());
    curtcReleaseLibrary(library.get());
    size_t count{};
    ASSERT_EQ(curtcEnumerateKernels(library.get(), &count, nullptr), CLANGRTC_SUCCESS);
    ASSERT_EQ(count, 1);
    curtcKernel info{};
    ASSERT_EQ(curtcEnumerateKernels(library.get(), &count, &info), CLANGRTC_SUCCESS);
    EXPECT_EQ(count, 1);
    EXPECT_EQ(test::to_string(info.id), "app-kernel");
    EXPECT_EQ(info.cudaName.size, 0);
    EXPECT_EQ(info.expression.size, 0);
    EXPECT_EQ(test::to_string(info.symbolName), "c_kernel");
    CUkernel handle{};
    ASSERT_EQ(curtcGetKernel(library.get(), info.symbolName, &handle), CLANGRTC_SUCCESS);
    ASSERT_NE(handle, nullptr);
    test::launch(handle, 42);
    library.reset();

    EXPECT_NO_THROW(curtcRetainLibrary(nullptr));
    EXPECT_NO_THROW(curtcReleaseLibrary(nullptr));
    EXPECT_NO_THROW(curtcDestroyCache(nullptr));
}

/**
 * @brief Enumerate selected metadata and retrieve already populated kernel handles.
 *
 * Checks selection aliases, count queries, insufficient and excess capacity, immutable output storage, and invalid
 * enumeration arguments.
 */
TEST(Curtc, EnumeratesMetadataAndRetrievesPrepopulatedKernels)
{
    test::TemporaryDirectory directory;
    auto owner = cache(directory);
    auto image = test::cubin_fixture();
    curtcInput input{};
    input.type = CURTC_INPUT_TYPE_CUBIN;
    input.data = {image.data(), image.size()};
    curtcKernelSelection entries[]{{inputs::to_view("first"), inputs::to_view("one")},
                                   {inputs::to_view("second"), inputs::to_view("two")},
                                   {inputs::to_view("first"), inputs::to_view("alias")}};
    auto r = request(input);
    input.entries = entries;
    input.entryCount = 3;

    auto library = get_library(owner.get(), r);

    size_t count = 99;
    ASSERT_EQ(curtcEnumerateKernels(library.get(), &count, nullptr), CLANGRTC_SUCCESS);
    ASSERT_EQ(count, 3);
    curtcKernel outputs[4];

    for (auto &output : outputs)
    {
        output = {inputs::to_view("unchanged"), inputs::to_view("untouched"), {}, {}};
    }

    for (size_t capacity : {0, 2})
    {
        count = capacity;
        EXPECT_EQ(curtcEnumerateKernels(library.get(), &count, outputs), CLANGRTC_INVALID_ARGUMENT);
        EXPECT_EQ(count, 3);

        for (auto const &output : outputs)
        {
            EXPECT_EQ(test::to_string(output.id), "unchanged");
            EXPECT_EQ(test::to_string(output.expression), "untouched");
        }
    }

    count = 4;
    ASSERT_EQ(curtcEnumerateKernels(library.get(), &count, outputs), CLANGRTC_SUCCESS);
    ASSERT_EQ(count, 3);
    EXPECT_EQ(test::to_string(outputs[3].id), "unchanged");
    EXPECT_EQ(test::to_string(outputs[3].expression), "untouched");

    for (size_t i = 0; i < count; ++i)
    {
        EXPECT_EQ(test::to_string(outputs[i].id), test::to_string(entries[i].id));
        EXPECT_EQ(outputs[i].expression.size, 0);
        EXPECT_EQ(test::to_string(outputs[i].symbolName), test::to_string(entries[i].name));
        EXPECT_EQ(outputs[i].cudaName.size, 0);
    }

    EXPECT_EQ(test::to_string(outputs[0].symbolName), test::to_string(outputs[2].symbolName));
    // Exact capacity succeeds too; enumeration never asks the driver to resolve a kernel.
    ASSERT_EQ(curtcEnumerateKernels(library.get(), &count, outputs), CLANGRTC_SUCCESS);
    EXPECT_EQ(count, 3);

    owner.reset();
    CUkernel kernel{};
    EXPECT_EQ(curtcGetKernel(library.get(), outputs[0].symbolName, &kernel), CLANGRTC_SUCCESS);
    test::launch(kernel, 42);

    EXPECT_EQ(test::to_string(outputs[0].id), "one");

    EXPECT_EQ(curtcEnumerateKernels(library.get(), nullptr, outputs), CLANGRTC_INVALID_ARGUMENT);
    EXPECT_EQ(curtcEnumerateKernels(library.get(), nullptr, nullptr), CLANGRTC_INVALID_ARGUMENT);
    EXPECT_EQ(curtcEnumerateKernels(nullptr, &count, outputs), CLANGRTC_INVALID_ARGUMENT);
    EXPECT_EQ(count, 0);
    EXPECT_EQ(test::to_string(outputs[0].id), "one");
    EXPECT_EQ(curtcEnumerateKernels(nullptr, &count, nullptr), CLANGRTC_INVALID_ARGUMENT);
    EXPECT_EQ(count, 0);
    EXPECT_EQ(curtcEnumerateKernels(nullptr, nullptr, nullptr), CLANGRTC_INVALID_ARGUMENT);
}

/**
 * @brief Reuse compiled code and a loaded library across identical requests.
 *
 * Tracks compilation, linking, and CUDA library reuse while retaining a library beyond cache and other reference
 * destruction.
 */
TEST(Curtc, ReusesCompilationAndLoadedKernelAndRetainsLibrary)
{
    test::TemporaryDirectory directory;
    auto owner = cache(directory);
    auto input = source();
    inputs::source_view view(input);
    auto r = request(view.value);

    auto first = get_library(owner.get(), r);
    auto before = statistics(owner.get());
    auto second = get_library(owner.get(), r);
    EXPECT_EQ(statistics(owner.get()).kernelCompilations, before.kernelCompilations);
    EXPECT_EQ(statistics(owner.get()).links, before.links);

    auto raw = kernel_handle(first.get(), 0);
    EXPECT_EQ(raw, kernel_handle(second.get(), 0));
    curtcRetainLibrary(first.get());
    auto retained = first.release();
    curtcReleaseLibrary(retained);
    second.reset();
    owner.reset();
    test::launch(raw, 42);
    curtcReleaseLibrary(retained);
}

/**
 * @brief Compile multiple template selections together and attribute work to the requesting call.
 *
 * Compares cold and warm per-call statistics, distinct kernel handles, shared reuse, and retained-library ownership.
 */
TEST(Curtc, OneRequestCompilesMultipleKernelsAndReportsItsOwnStatistics)
{
    test::TemporaryDirectory directory;
    auto owner = cache(directory);
    auto input = source();
    inputs::source_view view(input);
    auto r = request(view.value);
    curtcKernelSelection entries[]{{inputs::to_view("&fixed<int>"), {}}, {inputs::to_view("&fixed<float>"), {}}};
    view.value.entries = entries;
    view.value.entryCount = 2;
    curtcCompileStatistics cold{}, warm{};
    auto first = get_library(owner.get(), r, &cold);
    auto first_kernel = kernel_handle(first.get(), 0);
    auto second_kernel = kernel_handle(first.get(), 1);
    EXPECT_NE(first_kernel, nullptr);
    EXPECT_NE(second_kernel, nullptr);
    EXPECT_NE(first_kernel, second_kernel);
    EXPECT_EQ(cold.kernelCompilations, 1);
    EXPECT_EQ(cold.links, 1);
    EXPECT_EQ(cold.libraryLoads, 1);
    auto repeated = get_library(owner.get(), r, &warm);
    EXPECT_EQ(kernel_handle(repeated.get(), 0), first_kernel);
    EXPECT_EQ(kernel_handle(repeated.get(), 1), second_kernel);
    EXPECT_EQ(warm.kernelCompilations, 0);
    EXPECT_EQ(warm.fragmentCompilations, 0);
    EXPECT_EQ(warm.links, 0);
    EXPECT_EQ(warm.libraryLoads, 0);
    EXPECT_GE(warm.memoryHits, 3);
    EXPECT_GT(warm.totalNs, 0);
    curtcRetainLibrary(first.get());
    auto retained = first.release();
    curtcReleaseLibrary(retained);
    repeated.reset();
    owner.reset();
    test::launch(first_kernel, 42);
    test::launch(second_kernel, 42.0f);
    curtcReleaseLibrary(retained);
}

/**
 * @brief Reuse independently compiled inputs when finalizing their combined library.
 *
 * Finalizes each source separately, combines both without recompiling, then reverses input order to check metadata
 * ordering.
 */
TEST(Curtc, FinalizesIndividualInputsAndTheirCombinationWithIndependentCompilationCaches)
{
    test::TemporaryDirectory directory;
    auto owner = cache(directory);
    auto first_source = source();
    first_source.text = "extern \"C\" __global__ void first(int* out) { *out = 1; }";
    auto second_source = source();
    second_source.text = "extern \"C\" __global__ void second(int* out) { *out = 2; }";
    inputs::source_view first(first_source);
    inputs::source_view second(second_source);
    curtcKernelSelection entries[]{{inputs::to_view("&first"), inputs::to_view("one")},
                                   {inputs::to_view("&second"), inputs::to_view("two")}};
    curtcInput code[]{first.value, second.value};

    for (size_t i = 0; i < 2; ++i)
    {
        code[i].entries = &entries[i];
        code[i].entryCount = 1;
    }

    curtcRequest r{};
    r.target = inputs::to_view(test::gpu_target());
    r.inputCount = 1;

    for (auto const &input : code)
    {
        r.inputs = &input;
        curtcCompileStatistics stats{};
        auto library = get_library(owner.get(), r, &stats);
        EXPECT_NE(kernel_handle(library.get(), 0), nullptr);
        EXPECT_EQ(stats.kernelCompilations, 1);
        EXPECT_EQ(stats.links, 1);
        EXPECT_EQ(test::to_string(kernel_info(library.get(), 0).id), test::to_string(input.entries[0].id));
    }

    r.inputs = code;
    r.inputCount = 2;
    curtcCompileStatistics combined{}, warm{};
    auto library = get_library(owner.get(), r, &combined);
    auto repeated = get_library(owner.get(), r, &warm);
    EXPECT_EQ(combined.kernelCompilations, 0);
    EXPECT_EQ(combined.fragmentCompilations, 0);
    EXPECT_EQ(combined.links, 1);
    EXPECT_EQ(warm.links, 0);
    EXPECT_EQ(warm.libraryLoads, 0);
    EXPECT_EQ(test::to_string(kernel_info(library.get(), 0).symbolName), "first");
    EXPECT_EQ(test::to_string(kernel_info(library.get(), 1).symbolName), "second");
    EXPECT_EQ(test::to_string(kernel_info(library.get(), 1).expression), "&second");
    EXPECT_EQ(test::to_string(kernel_info(library.get(), 1).cudaName), "second");
    EXPECT_EQ(kernel_handle(library.get(), 1), kernel_handle(repeated.get(), 1));

    std::swap(code[0], code[1]);
    auto reversed = get_library(owner.get(), r);
    EXPECT_EQ(test::to_string(kernel_info(reversed.get(), 0).id), "two");
    EXPECT_EQ(test::to_string(kernel_info(reversed.get(), 1).id), "one");
}

/**
 * @brief Select kernels from either or both inputs across source, artifact, LTO, and PTX combinations.
 *
 * Checks per-input identities and source metadata for each selection mask and verifies warm calls perform no new
 * compilation, linking, or loading.
 */
TEST(Curtc, SelectsKernelsFromEitherOrBothSourceArtifactAndBinaryInputs)
{
    auto first_source = source();
    first_source.text = "extern \"C\" __global__ void first(int* out) { *out = 1; }";
    auto second_source = source();
    second_source.text = "extern \"C\" __global__ void second(int* out) { *out = 2; }";
    inputs::source_view first(first_source);
    inputs::source_view second(second_source);
    curtcKernelSelection entries[]{{inputs::to_view("&first"), inputs::to_view("one")}, {{}, inputs::to_view("two")}};

    for (auto type : {CURTC_INPUT_TYPE_CUDA, CURTC_INPUT_TYPE_ARTIFACT, CURTC_INPUT_TYPE_LTO, CURTC_INPUT_TYPE_PTX})
    {
        SCOPED_TRACE(static_cast<int>(type));
        test::TemporaryDirectory directory;
        auto owner = cache(directory);
        curtcInput code[]{first.value, second.value};
        code[1].type = type;
        bool source_input = type == CURTC_INPUT_TYPE_CUDA || type == CURTC_INPUT_TYPE_ARTIFACT;
        entries[1].name = inputs::to_view(source_input ? "&second" : "second");
        std::string bytes;
        test::LoadedArtifact environment{CLANGRTC_SUCCESS, test::Artifact(nullptr, clangrtcDestroyArtifact)};

        if (type != CURTC_INPUT_TYPE_CUDA)
        {
            auto flags = second_source.options;
            flags.push_back(std::format("--gpu-architecture={}", test::gpu_target()));

            switch (type)
            {
            case CURTC_INPUT_TYPE_ARTIFACT:
                flags.push_back("--artifact=pch");
                break;

            case CURTC_INPUT_TYPE_LTO:
                flags.push_back("--lto-ir");
                break;

            default:
                flags.push_back("--ptx-text");
                break;
            }

            auto input = test::request(second_source.text, {});
            auto compiled =
                type == CURTC_INPUT_TYPE_ARTIFACT ? test::prepare(input, flags) : test::compile(input, flags);
            ASSERT_EQ(compiled.status, CLANGRTC_SUCCESS) << compiled.log();
            bytes = compiled.output();
            code[1].data = inputs::to_view(bytes);
            code[1].source = {};

            if (type == CURTC_INPUT_TYPE_ARTIFACT)
            {
                environment = test::load(bytes);
                ASSERT_EQ(environment.status, CLANGRTC_SUCCESS);
                code[1].source.artifact = environment.artifact.get();
            }
            code[1].options = nullptr;
            code[1].optionCount = 0;
        }

        curtcRequest r{};
        r.inputs = code;
        r.inputCount = 2;
        r.target = inputs::to_view(test::gpu_target());

        for (unsigned mask : {1U, 2U, 3U})
        {
            SCOPED_TRACE(mask);

            for (unsigned i = 0; i < 2; ++i)
            {
                code[i].entries = mask & (1U << i) ? &entries[i] : nullptr;
                code[i].entryCount = mask & (1U << i) ? 1 : 0;
            }

            auto library = get_library(owner.get(), r);
            size_t count{};
            ASSERT_EQ(curtcEnumerateKernels(library.get(), &count, nullptr), CLANGRTC_SUCCESS);
            ASSERT_EQ(count, mask == 3 ? 2 : 1);
            size_t selected{};

            for (size_t i = 0; i < 2; ++i)
            {
                if (!code[i].entryCount)
                {
                    continue;
                }

                auto info = kernel_info(library.get(), selected++);
                EXPECT_EQ(test::to_string(info.id), i == 0 ? "one" : "two");
                EXPECT_EQ(test::to_string(info.symbolName), i == 0 ? "first" : "second");
                bool has_source = i == 0 || source_input;
                std::string expected_name = i == 0 ? "first" : "second";
                EXPECT_EQ(test::to_string(info.expression), has_source ? std::format("&{}", expected_name) : "");
                EXPECT_EQ(test::to_string(info.cudaName), has_source ? expected_name : "");
            }

            curtcCompileStatistics warm{};
            auto repeated = get_library(owner.get(), r, &warm);
            EXPECT_EQ(warm.kernelCompilations, 0);
            EXPECT_EQ(warm.fragmentCompilations, 0);
            EXPECT_EQ(warm.links, 0);
            EXPECT_EQ(warm.libraryLoads, 0);
        }
    }
}

/**
 * @brief Keep retained definitions separate from selected-kernel metadata and cache identity.
 *
 * Retains a device specialization alongside kernels, then changes which expressions are kernel selections without
 * changing the full expression list.
 */
TEST(Curtc, RetainsDefinitionsAlongsideKernelSelectionsWithoutMixingTheirCachedMetadata)
{
    test::TemporaryDirectory directory;
    auto owner = cache(directory);
    auto input = source();
    input.text.append("\ntemplate<class T> __device__ T retained(T x) { return x; }\n");
    input.expressions = {"&retained<double>"};
    inputs::source_view view(input);
    auto r = request(view.value);
    auto library = get_library(owner.get(), r);
    size_t count{};
    ASSERT_EQ(curtcEnumerateKernels(library.get(), &count, nullptr), CLANGRTC_SUCCESS);
    EXPECT_EQ(count, 1);
    EXPECT_EQ(test::to_string(kernel_info(library.get(), 0).expression), "&fixed<int>");

    clangrtcString retained{inputs::to_view("&fixed<int>")};
    view.value.source.expressions = &retained;
    curtcKernelSelection entries[]{{inputs::to_view("&fixed<int>"), {}}, {inputs::to_view("&fixed<float>"), {}}};
    view.value.entries = &entries[1];
    auto one = get_library(owner.get(), r);
    EXPECT_EQ(test::to_string(kernel_info(one.get(), 0).cudaName), "fixed<float>");

    // The full expression list is unchanged, but both expressions now select kernels.
    view.value.source.expressions = nullptr;
    view.value.source.expressionCount = 0;
    view.value.entries = entries;
    view.value.entryCount = 2;
    auto both = get_library(owner.get(), r);
    ASSERT_EQ(curtcEnumerateKernels(both.get(), &count, nullptr), CLANGRTC_SUCCESS);
    EXPECT_EQ(count, 2);
    EXPECT_EQ(test::to_string(kernel_info(both.get(), 0).cudaName), "fixed<int>");
    EXPECT_EQ(test::to_string(kernel_info(both.get(), 1).cudaName), "fixed<float>");
}

/**
 * @brief Reject a valid device-function expression when supplied as a kernel selection.
 *
 * Checks invalid-expression status, a cleared library output, and a diagnostic identifying that the selection is not a
 * CUDA kernel.
 */
TEST(Curtc, RejectsDeviceFunctionsSelectedAsKernels)
{
    test::TemporaryDirectory directory;
    auto owner = cache(directory);
    auto input = source();
    input.text = "extern \"C\" __device__ int callable(int x) { return x; }";
    inputs::source_view view(input);
    auto r = request(view.value);
    curtcKernelSelection entry{inputs::to_view("&callable"), {}};
    view.value.entries = &entry;
    curtcLibrary output = reinterpret_cast<curtcLibrary>(1);
    diagnostic log;
    EXPECT_EQ(curtcGet(owner.get(), &r, diagnostic::collect, &log, &output, nullptr), CLANGRTC_INVALID_EXPRESSION);
    EXPECT_EQ(output, nullptr);
    EXPECT_NE(log.message.find("not a CUDA kernel"), std::string::npos);
}

/**
 * @brief Allow code inputs without selected metadata while preserving exported-kernel lookup.
 *
 * Compiles an unselected visible kernel, checks empty enumeration, and retrieves the kernel by its exact symbol.
 */
TEST(Curtc, LoadsInputsWithoutSelectedMetadataAndRetrievesExportedKernelsBySymbol)
{
    test::TemporaryDirectory directory;
    auto owner = cache(directory);
    auto input = source();
    input.text = "extern \"C\" __global__ void visible(int* out) { *out = 42; }";
    inputs::source_view view(input);
    curtcRequest r{};
    r.inputs = &view.value;
    r.inputCount = 1;
    r.target = inputs::to_view(test::gpu_target());
    curtcCompileStatistics stats{};
    auto library = get_library(owner.get(), r, &stats);
    size_t count = 99;
    EXPECT_EQ(curtcEnumerateKernels(library.get(), &count, nullptr), CLANGRTC_SUCCESS);
    EXPECT_EQ(count, 0);
    CUkernel kernel{};
    EXPECT_EQ(curtcGetKernel(library.get(), inputs::to_view("visible"), &kernel), CLANGRTC_SUCCESS);
    EXPECT_NE(kernel, nullptr);
    EXPECT_EQ(stats.fragmentCompilations, 1);
    EXPECT_EQ(stats.kernelCompilations, 0);
}

/**
 * @brief Validate opaque binary selections against the complete loaded export table.
 *
 * Checks missing-symbol failures, retry without another library load, access to unselected exports, and binary metadata
 * identities.
 */
TEST(Curtc, BinarySelectionsAreValidatedAgainstThePrepopulatedLibrary)
{
    test::TemporaryDirectory directory;
    auto owner = cache(directory);
    auto code = test::cubin_fixture();
    curtcInput input{};
    input.type = CURTC_INPUT_TYPE_CUBIN;
    input.data = {code.data(), code.size()};
    auto r = request(input);
    curtcKernelSelection entries[]{{inputs::to_view("first"), inputs::to_view("aot-first")},
                                   {inputs::to_view("missing"), inputs::to_view("aot-second")}};
    input.entries = entries;
    input.entryCount = 2;
    curtcCompileStatistics stats{};
    auto output = reinterpret_cast<curtcLibrary>(1);
    EXPECT_EQ(curtcGet(owner.get(), &r, nullptr, nullptr, &output, &stats), CLANGRTC_TOOLCHAIN_ERROR);
    EXPECT_EQ(output, nullptr);
    EXPECT_EQ(stats.libraryLoads, 1);

    EXPECT_EQ(curtcGet(owner.get(), &r, nullptr, nullptr, &output, &stats), CLANGRTC_TOOLCHAIN_ERROR);
    EXPECT_EQ(output, nullptr);
    EXPECT_EQ(stats.libraryLoads, 0);

    entries[1].name = inputs::to_view("second");
    auto loaded = get_library(owner.get(), r, &stats);
    EXPECT_EQ(stats.libraryLoads, 0);
    EXPECT_NE(kernel_handle(loaded.get(), 0), kernel_handle(loaded.get(), 1));
    CUkernel unselected{};
    EXPECT_EQ(curtcGetKernel(loaded.get(), inputs::to_view("fixed"), &unselected), CLANGRTC_SUCCESS);
    EXPECT_NE(unselected, nullptr);
    auto missing = reinterpret_cast<CUkernel>(1);
    EXPECT_EQ(curtcGetKernel(loaded.get(), inputs::to_view("missing"), &missing), CLANGRTC_TOOLCHAIN_ERROR);
    EXPECT_EQ(missing, nullptr);

    auto info = kernel_info(loaded.get(), 1);
    EXPECT_EQ(test::to_string(info.id), "aot-second");
    EXPECT_EQ(test::to_string(info.symbolName), "second");
}

/**
 * @brief Preload all exported kernels and reuse a loaded library across different selections.
 *
 * Loads PTX with more exports than selected metadata and checks that later selections reuse the same loaded library
 * and prepopulated kernel handles.
 */
TEST(Curtc, PreloadsEveryBinaryExportAndSharesLibrariesAcrossSelections)
{
    auto flags = source().options;
    flags.insert(flags.end(), {std::format("--gpu-architecture={}", test::gpu_target()), "--ptx-text"});
    auto compiled = test::compile(test::request(R"CUDA(
                                                    extern "C" __global__ void first()
                                                    {
                                                    }

                                                    extern "C" __global__ void second()
                                                    {
                                                    }
                                                )CUDA",
                                                {}),
                                  flags);
    ASSERT_EQ(compiled.status, CLANGRTC_SUCCESS) << compiled.log();
    auto ptx = compiled.output();
    curtcInput input{};
    input.type = CURTC_INPUT_TYPE_PTX;
    input.data = inputs::to_view(ptx);
    auto r = request(input);
    curtcKernelSelection entry{inputs::to_view("first"), inputs::to_view("one")};
    input.entries = &entry;
    test::TemporaryDirectory directory;
    auto owner = cache(directory);

    auto first = get_library(owner.get(), r);

    CUkernel unselected{};
    EXPECT_EQ(curtcGetKernel(first.get(), inputs::to_view("second"), &unselected), CLANGRTC_SUCCESS);
    EXPECT_NE(unselected, nullptr);
    entry = {inputs::to_view("second"), inputs::to_view("two")};
    curtcCompileStatistics reused{};
    auto second = get_library(owner.get(), r, &reused);
    EXPECT_EQ(kernel_handle(second.get(), 0), unselected);
    EXPECT_EQ(reused.libraryLoads, 0);
    EXPECT_EQ(reused.links, 0);

    EXPECT_EQ(test::to_string(kernel_info(first.get(), 0).id), "one");
    EXPECT_EQ(test::to_string(kernel_info(second.get(), 0).id), "two");
}

/**
 * @brief Validate kernel-lookup arguments and clear a provided output on failure.
 *
 * Covers invalid library and output handles, empty or NUL-containing names, and unknown exported symbols.
 */
TEST(Curtc, KernelLookupValidatesSymbolsAndClearsOutputs)
{
    test::TemporaryDirectory directory;
    auto owner = cache(directory);
    auto image = test::cubin_fixture();
    curtcInput input{};
    input.type = CURTC_INPUT_TYPE_CUBIN;
    input.data = {image.data(), image.size()};
    auto library = get_library(owner.get(), request(input));

    std::string embedded_null("first\0second", 12);

    for (auto symbol : {clangrtcString{}, inputs::to_view(embedded_null)})
    {
        auto output = reinterpret_cast<CUkernel>(1);
        EXPECT_EQ(curtcGetKernel(library.get(), symbol, &output), CLANGRTC_INVALID_ARGUMENT);
        EXPECT_EQ(output, nullptr);
    }

    auto output = reinterpret_cast<CUkernel>(1);
    EXPECT_EQ(curtcGetKernel(nullptr, inputs::to_view("first"), &output), CLANGRTC_INVALID_ARGUMENT);
    EXPECT_EQ(output, nullptr);
    EXPECT_EQ(curtcGetKernel(library.get(), inputs::to_view("first"), nullptr), CLANGRTC_INVALID_ARGUMENT);
}

/**
 * @brief Own per-request identities and source metadata independently of input storage and cached code.
 *
 * Exercises template aliases, modified IDs, retained libraries, and cache reuse while preserving each result metadata
 * snapshot.
 */
TEST(Curtc, KernelMetadataSurvivesLtoCachesAndRequestStorage)
{
    test::TemporaryDirectory directory;
    Library retained(nullptr, curtcReleaseLibrary);
    {
        auto owner = cache(directory);
        auto input = source();
        input.text = "namespace demo { template<class T> __global__ void fixed(T* out) { *out = T{42}; } }";
        inputs::source_view view(input);
        auto r = request(view.value);
        std::string id = "original";
        std::string expression = "&demo::fixed<int>";
        // Two selectors resolve to the same declaration without duplicating its source name.
        curtcKernelSelection entries[]{{inputs::to_view(expression), inputs::to_view(id)},
                                       {inputs::to_view("(&demo::fixed<int>)"), inputs::to_view("alias")},
                                       {inputs::to_view("&demo::fixed<float>"), {}}};
        view.value.entries = entries;
        view.value.entryCount = 3;
        retained = get_library(owner.get(), r);
        auto original = kernel_info(retained.get(), 0);
        EXPECT_EQ(test::to_string(original.id), "original");
        EXPECT_EQ(test::to_string(original.expression), "&demo::fixed<int>");
        EXPECT_EQ(test::to_string(original.cudaName), "demo::fixed<int>");
        EXPECT_EQ(test::to_string(original.symbolName), "_ZN4demo5fixedIiEEvPT_");
        auto alias = kernel_info(retained.get(), 1);
        EXPECT_EQ(test::to_string(alias.id), "alias");
        EXPECT_EQ(test::to_string(alias.expression), "(&demo::fixed<int>)");
        EXPECT_EQ(test::to_string(alias.cudaName), test::to_string(original.cudaName));
        EXPECT_EQ(test::to_string(alias.symbolName), test::to_string(original.symbolName));
        auto other = kernel_info(retained.get(), 2);
        EXPECT_EQ(other.id.size, 0);
        EXPECT_EQ(test::to_string(other.cudaName), "demo::fixed<float>");
        EXPECT_EQ(test::to_string(other.expression), "&demo::fixed<float>");

        id.assign("renamed");
        entries[0].id = inputs::to_view(id);
        curtcCompileStatistics warm{};
        auto reused = get_library(owner.get(), r, &warm);
        EXPECT_EQ(test::to_string(kernel_info(reused.get(), 0).id), "renamed");
        EXPECT_EQ(test::to_string(kernel_info(reused.get(), 0).expression), expression);
        EXPECT_EQ(test::to_string(original.id), "original");
        EXPECT_EQ(kernel_handle(retained.get(), 0), kernel_handle(reused.get(), 0));
        EXPECT_EQ(warm.kernelCompilations, 0);
        EXPECT_EQ(warm.links, 0);
        EXPECT_EQ(warm.libraryLoads, 0);

        reused.reset();
        owner = cache(directory);
        entries[0].id = inputs::to_view("disk");
        curtcCompileStatistics disk{};
        auto restored = get_library(owner.get(), r, &disk);
        auto info = kernel_info(restored.get(), 0);
        EXPECT_EQ(test::to_string(info.id), "disk");
        EXPECT_EQ(test::to_string(info.expression), expression);
        EXPECT_EQ(test::to_string(info.cudaName), test::to_string(original.cudaName));
        EXPECT_EQ(test::to_string(info.symbolName), test::to_string(original.symbolName));
        EXPECT_GE(disk.diskHits, 2);
        EXPECT_EQ(disk.kernelCompilations, 0);
        EXPECT_EQ(disk.links, 0);
    }

    auto info = kernel_info(retained.get(), 0);
    EXPECT_EQ(test::to_string(info.id), "original");
    EXPECT_EQ(test::to_string(info.expression), "&demo::fixed<int>");
    EXPECT_EQ(test::to_string(info.cudaName), "demo::fixed<int>");
    EXPECT_EQ(test::to_string(info.symbolName), "_ZN4demo5fixedIiEEvPT_");
}

/**
 * @brief Finalize selected source kernels without requiring unrelated unresolved kernels.
 *
 * Adds an unselected kernel calling a missing device function and verifies that the requested specialization remains
 * usable.
 */
TEST(Curtc, SourceSelectionExcludesUnrelatedUnresolvedKernels)
{
    test::TemporaryDirectory directory;
    auto owner = cache(directory);
    auto input = source();
    input.text.append("\nextern \"C\" __device__ int missing_udf(int);\n"
                      "extern \"C\" __global__ void unrelated(int* out) { *out = missing_udf(20); }");
    inputs::source_view view(input);
    auto r = request(view.value);
    EXPECT_NE(kernel_handle(get_library(owner.get(), r).get(), 0), nullptr);
}

/**
 * @brief Reuse persistent compilation and link results across independent cache objects.
 *
 * Creates a second cache on the same directory and checks disk hits with no new compilation or linking, while loaded
 * libraries remain cache-owned separately.
 */
TEST(Curtc, CacheInstancesHaveIndependentPolicyAndReuseDisk)
{
    test::TemporaryDirectory directory;
    auto input = source();
    inputs::source_view view(input);
    auto r = request(view.value);
    auto first = cache(directory);
    auto loaded = get_library(first.get(), r);
    EXPECT_EQ(statistics(first.get()).kernelCompilations, 1);
    auto second = cache(directory);
    EXPECT_EQ(statistics(second.get()).kernelCompilations, 0);
    auto restored = get_library(second.get(), r);
    auto stats = statistics(second.get());
    EXPECT_EQ(stats.kernelCompilations, 0);
    EXPECT_EQ(stats.links, 0);
    EXPECT_GE(stats.diskHits, 2);
    EXPECT_NE(kernel_handle(loaded.get(), 0), kernel_handle(restored.get(), 0));
}

/** @brief Use immutable artifacts' precomputed identities when selecting different prepared programs. */
TEST(Curtc, ReadyArtifactIdentityDistinguishesDifferentPreparedPrograms)
{
    auto input = source();
    auto flags = input.options;
    flags.insert(flags.end(), {std::format("--gpu-architecture={}", test::gpu_target()), "--artifact=pch"});
    auto first = test::prepare(test::request(input.text, {}), flags);
    ASSERT_EQ(first.status, CLANGRTC_SUCCESS) << first.log();
    auto first_package = first.output();
    auto position = input.text.find("42");
    ASSERT_NE(position, std::string::npos);
    input.text.replace(position, 2, "43");
    auto second = test::prepare(test::request(input.text, {}), flags);
    ASSERT_EQ(second.status, CLANGRTC_SUCCESS) << second.log();
    auto second_package = second.output();
    auto first_environment = test::load(first_package);
    auto second_environment = test::load(second_package);
    ASSERT_EQ(first_environment.status, CLANGRTC_SUCCESS);
    ASSERT_EQ(second_environment.status, CLANGRTC_SUCCESS);
    curtcInput code{};
    code.type = CURTC_INPUT_TYPE_ARTIFACT;
    code.source.artifact = first_environment.artifact.get();
    auto r = request(code);
    test::TemporaryDirectory directory;
    auto owner = cache(directory);
    auto original = get_library(owner.get(), r);
    test::launch(kernel_handle(original.get(), 0), 42);
    code.source.artifact = second_environment.artifact.get();
    curtcCompileStatistics changed{};
    auto replaced = get_library(owner.get(), r, &changed);
    EXPECT_EQ(changed.kernelCompilations, 1);
    EXPECT_EQ(changed.links, 1);
    test::launch(kernel_handle(replaced.get(), 0), 43);
    test::launch(kernel_handle(original.get(), 0), 42);
    curtcCompileStatistics repeated{};
    auto reused = get_library(owner.get(), r, &repeated);
    EXPECT_EQ(repeated.kernelCompilations, 0);
    EXPECT_EQ(repeated.links, 0);
    test::launch(kernel_handle(reused.get(), 0), 43);
}

/**
 * @brief Share identical concurrent requests through compilation, finalization, and CUDA loading.
 *
 * Starts overlapping callers and reconciles their per-call statistics with aggregate counters and real CUDA loading
 * statistics.
 */
TEST(Curtc, ConcurrentRequestsShareCompilationLinkingAndLoading)
{
    test::TemporaryDirectory directory;
    auto owner = cache(directory);
    auto input = source();
    inputs::source_view view(input);
    auto r = request(view.value);

    std::latch start(8);
    std::vector<std::future<Library>> jobs;
    std::vector<curtcCompileStatistics> per_call(8);

    for (int i = 0; i < 8; ++i)
    {
        jobs.push_back(std::async(std::launch::async, [&, i] {
            start.count_down();
            start.wait();
            auto result = get_library(owner.get(), r, &per_call[i]);
            EXPECT_NE(kernel_handle(result.get(), 0), nullptr);
            return result;
        }));
    }

    std::vector<Library> kernels;

    for (auto &job : jobs)
    {
        kernels.push_back(job.get());
        EXPECT_EQ(kernel_handle(kernels.back().get(), 0), kernel_handle(kernels.front().get(), 0));
    }

    auto stats = statistics(owner.get());
    EXPECT_EQ(stats.kernelCompilations, 1);
    EXPECT_EQ(stats.links, 1);

    curtcCompileStatistics sum{};

    for (auto const &item : per_call)
    {
        sum.kernelCompilations += item.kernelCompilations;
        sum.links += item.links;
        sum.libraryLoads += item.libraryLoads;
        sum.memoryHits += item.memoryHits;
        sum.diskHits += item.diskHits;
        sum.misses += item.misses;
        sum.sharedRequests += item.sharedRequests;
        EXPECT_GT(item.totalNs, 0);
    }

    EXPECT_EQ(sum.kernelCompilations, 1);
    EXPECT_EQ(sum.links, 1);
    EXPECT_EQ(sum.libraryLoads, 1);
    EXPECT_EQ(sum.memoryHits, stats.memoryHits);
    EXPECT_EQ(sum.diskHits, stats.diskHits);
    EXPECT_EQ(sum.misses, stats.misses);
    EXPECT_EQ(sum.sharedRequests, stats.sharedRequests);
}

/**
 * @brief Link a prepared artifact consumer against source and precompiled device implementations.
 *
 * Uses the same selected kernel with a CUDA implementation and then its LTO equivalent, checking function-input
 * compilation accounting.
 */
TEST(Curtc, LinksSourceAndBinaryFragmentsWithPreparedFrontend)
{
    test::TemporaryDirectory directory;
    auto owner = cache(directory);
    auto input = source();
    input.text.append("\nextern \"C\" __device__ int udf(int);\n"
                      "extern \"C\" __global__ void staged(int* out) { *out = udf(20); }");
    auto flags = input.options;
    flags.insert(flags.end(), {std::format("--gpu-architecture={}", test::gpu_target()), "--artifact=pch"});
    auto prepared = test::prepare(test::request(input.text, {}), flags);
    ASSERT_EQ(prepared.status, CLANGRTC_SUCCESS) << prepared.log();
    auto package = prepared.output();
    auto environment = test::load(package);
    ASSERT_EQ(environment.status, CLANGRTC_SUCCESS);
    curtcInput base{};
    base.type = CURTC_INPUT_TYPE_ARTIFACT;
    base.source.artifact = environment.artifact.get();
    auto r = request(base);
    curtcKernelSelection selected{inputs::to_view("&staged"), {}};
    base.entries = &selected;
    base.entryCount = 1;
    inputs::Source udf;
    udf.text = "extern \"C\" __device__ int udf(int x) { return x*2; }";
    udf.options = input.options;
    inputs::source_view fragment(udf);
    curtcInput combined[]{base, fragment.value};
    r.inputs = combined;
    r.inputCount = 2;
    auto linked = get_library(owner.get(), r);
    EXPECT_NE(kernel_handle(linked.get(), 0), nullptr);
    EXPECT_EQ(statistics(owner.get()).fragmentCompilations, 1);
    flags = udf.options;
    flags.insert(flags.end(), {std::format("--gpu-architecture={}", test::gpu_target()), "--lto-ir"});
    auto binary = test::compile(test::request(udf.text, {}), flags);
    ASSERT_EQ(binary.status, CLANGRTC_SUCCESS) << binary.log();
    auto code = binary.output();
    curtcInput precompiled{};
    precompiled.type = CURTC_INPUT_TYPE_LTO;
    precompiled.data = inputs::to_view(code);
    combined[1] = precompiled;
    EXPECT_NE(kernel_handle(get_library(owner.get(), r).get(), 0), nullptr);
}

/**
 * @brief Retry failed finalization while reusing the successfully compiled kernel LTO.
 *
 * Leaves a device function unresolved and checks nvJitLink diagnostics, empty output, repeated link attempts, and no
 * second compilation.
 */
TEST(Curtc, LinkFailuresPreserveDiagnosticsAndRetryWithCachedLto)
{
    test::TemporaryDirectory directory;
    auto owner = cache(directory);
    auto input = source();
    input.text = "extern \"C\" __device__ int missing_for_link(int);\n"
                 "extern \"C\" __global__ void failed(int* out) { *out = missing_for_link(1); }";
    inputs::source_view view(input);
    auto r = request(view.value);
    curtcKernelSelection selected{inputs::to_view("&failed"), {}};
    view.value.entries = &selected;

    for (int attempt = 0; attempt < 2; ++attempt)
    {
        diagnostic log;
        auto output = reinterpret_cast<curtcLibrary>(1);
        curtcCompileStatistics stats{};
        EXPECT_EQ(curtcGet(owner.get(), &r, diagnostic::collect, &log, &output, &stats), CLANGRTC_LINK_FAILED);
        EXPECT_EQ(output, nullptr);
        EXPECT_EQ(log.stage, "nvjitlink");
        EXPECT_NE(log.message.find("nvJitLinkComplete"), std::string::npos);
        EXPECT_NE(log.message.find("curtc.cpp:"), std::string::npos);
        EXPECT_NE(log.message.find("missing_for_link"), std::string::npos);
        EXPECT_EQ(stats.kernelCompilations, attempt == 0 ? 1 : 0);
        EXPECT_EQ(stats.links, 1);
    }

    EXPECT_EQ(statistics(owner.get()).links, 2);
}

/**
 * @brief Finalize compatible precompiled LTO or PTX inputs without CUDA source.
 *
 * Checks opaque kernel metadata, repeated-request reuse, and identity changes that do not trigger another link or
 * library load.
 */
TEST(Curtc, LinksOnlyPrecompiledFragmentsWithoutCudaSource)
{
    curtcKernelSelection selected{};
    test::TemporaryDirectory directory;
    auto owner = cache(directory);

    for (auto type : {CURTC_INPUT_TYPE_LTO, CURTC_INPUT_TYPE_PTX})
    {
        auto flags = source().options;
        flags.insert(flags.end(), {std::format("--gpu-architecture={}", test::gpu_target()),
                                   type == CURTC_INPUT_TYPE_LTO ? "--lto-ir" : "--ptx-text"});
        auto kernel = test::compile(test::request("extern \"C\" __device__ int udf(int); "
                                                  "extern \"C\" __global__ void staged(int* out) { *out = udf(20); }",
                                                  {}),
                                    flags);
        auto udf = test::compile(test::request("extern \"C\" __device__ int udf(int x) { return x*2; }", {}), flags);
        ASSERT_EQ(kernel.status, CLANGRTC_SUCCESS) << kernel.log();
        ASSERT_EQ(udf.status, CLANGRTC_SUCCESS) << udf.log();
        auto kernel_bytes = kernel.output(), udf_bytes = udf.output();
        curtcInput fragments[2]{};
        fragments[0].type = type;
        fragments[0].data = inputs::to_view(kernel_bytes);
        fragments[1].type = type;
        fragments[1].data = inputs::to_view(udf_bytes);
        curtcRequest r{};
        selected.name = inputs::to_view("staged");
        selected.id = inputs::to_view("opaque-kernel");
        fragments[0].entries = &selected;
        fragments[0].entryCount = 1;
        r.target = inputs::to_view(test::gpu_target());
        r.inputs = fragments;
        r.inputCount = 2;
        auto before = statistics(owner.get());
        auto linked = get_library(owner.get(), r);
        auto repeated = get_library(owner.get(), r);
        EXPECT_EQ(kernel_handle(linked.get(), 0), kernel_handle(repeated.get(), 0));
        auto info = kernel_info(linked.get(), 0);
        EXPECT_EQ(test::to_string(info.id), "opaque-kernel");
        EXPECT_EQ(info.expression.size, 0);
        EXPECT_EQ(info.cudaName.size, 0);
        EXPECT_EQ(test::to_string(info.symbolName), "staged");
        // Application identities do not affect compilation, linking, or CUDA library reuse.
        selected.id = inputs::to_view("renamed-kernel");
        curtcCompileStatistics reused{};
        auto renamed = get_library(owner.get(), r, &reused);
        auto renamed_info = kernel_info(renamed.get(), 0);
        EXPECT_EQ(test::to_string(renamed_info.id), "renamed-kernel");
        EXPECT_EQ(renamed_info.expression.size, 0);
        EXPECT_EQ(renamed_info.cudaName.size, 0);
        EXPECT_EQ(test::to_string(renamed_info.symbolName), "staged");
        EXPECT_EQ(test::to_string(info.id), "opaque-kernel");
        EXPECT_EQ(kernel_handle(renamed.get(), 0), kernel_handle(linked.get(), 0));
        EXPECT_EQ(reused.links, 0);
        EXPECT_EQ(reused.libraryLoads, 0);
        auto after = statistics(owner.get());
        EXPECT_EQ(after.kernelCompilations, before.kernelCompilations);
        EXPECT_EQ(after.fragmentCompilations, before.fragmentCompilations);
        EXPECT_EQ(after.links, before.links + 1);
    }
}

/**
 * @brief Keep caller-held CUDA libraries alive after their cache entries are evicted.
 *
 * Uses a tiny memory budget and two real cubins, then executes caller-owned kernels after eviction and cache
 * destruction.
 */
TEST(Curtc, EvictionPreservesCallerOwnedCudaLibraries)
{
    curtcKernelSelection selected{};
    curtcConfiguration config{};
    ASSERT_EQ(curtcDefaultConfiguration(&config), CLANGRTC_SUCCESS);
    config.diskCache = false;
    config.memoryBytes = 5;
    curtcCache output{};
    ASSERT_EQ(curtcCreateCache(&config, &output), CLANGRTC_SUCCESS);
    Cache owner(output, curtcDestroyCache);
    auto image = test::cubin_fixture(42);
    curtcInput input{};
    input.type = CURTC_INPUT_TYPE_CUBIN;
    input.data = {image.data(), image.size()};
    auto r = request(input);
    selected.name = inputs::to_view("first");
    input.entries = &selected;
    input.entryCount = 1;
    auto first = get_library(owner.get(), r);
    auto raw = kernel_handle(first.get(), 0);
    auto other = test::cubin_fixture(43);
    input.data = {other.data(), other.size()};
    selected.name = inputs::to_view("second");
    input.entries = &selected;
    input.entryCount = 1;
    auto second = get_library(owner.get(), r);
    test::launch(raw, 42);
    first.reset();

    auto retained = kernel_handle(second.get(), 0);
    owner.reset();
    test::launch(retained, 43);
    second.reset();
}

class CurtcFrontend : public ::testing::TestWithParam<char const *>
{
};

/**
 * @brief Instantiate several kernel types from a packaged frontend after deleting its headers.
 *
 * Runs PCH, BMI, and reduced-BMI consumers and checks resolved expressions, CUDA source names, and kernel handles.
 */
TEST_P(CurtcFrontend, ReusesPackagedHeadersForMultipleRuntimeInstantiations)
{
    test::TemporaryDirectory source_directory;
    auto header = source_directory.path / "factor.cuh";
    {
        std::ofstream file(header);
        file << "#define FACTOR 7\n";
    }

    bool module = std::string_view(GetParam()) != "pch";
    std::string prefix = module ? "module;\n#include <clangrtc/device.hpp>\n" : "";
    prefix.append(std::format("#include \"{}\"\n", header.string()));

    if (module)
    {
        prefix.append("export module curtc_frontend;\nexport ");
    }

    prefix.append("template<class T> __global__ void scaled(T* out) { *out = T{FACTOR}; }");
    auto flags = source().options;
    flags.insert(flags.end(),
                 {std::format("--gpu-architecture={}", test::gpu_target()), std::format("--artifact={}", GetParam())});
    auto prepared = test::prepare(test::request(prefix, {}), flags);
    ASSERT_EQ(prepared.status, CLANGRTC_SUCCESS) << prepared.log();
    auto package = prepared.output();
    auto environment = test::load(package);
    ASSERT_EQ(environment.status, CLANGRTC_SUCCESS);
    std::filesystem::remove_all(source_directory.path);
    test::TemporaryDirectory directory;
    auto owner = cache(directory);
    curtcInput input{};
    input.type = CURTC_INPUT_TYPE_ARTIFACT;
    input.source.artifact = environment.artifact.get();
    auto r = request(input);
    curtcKernelSelection entries[]{{inputs::to_view("&scaled<int>"), {}}, {inputs::to_view("&scaled<double>"), {}}};
    input.entries = entries;
    input.entryCount = 2;
    auto library = get_library(owner.get(), r);
    EXPECT_NE(kernel_handle(library.get(), 0), nullptr);
    EXPECT_NE(kernel_handle(library.get(), 1), nullptr);
    EXPECT_EQ(test::to_string(kernel_info(library.get(), 0).cudaName), "scaled<int>");
    EXPECT_EQ(test::to_string(kernel_info(library.get(), 1).cudaName), "scaled<double>");
    EXPECT_EQ(test::to_string(kernel_info(library.get(), 0).expression), "&scaled<int>");
    EXPECT_EQ(test::to_string(kernel_info(library.get(), 1).expression), "&scaled<double>");
}

/**
 * @brief Preserve explicit artifact-consumer source, virtual headers, filenames, and backend options.
 *
 * Preparation selects textual IR, which must not conflict with curtc's LTO output. First forces a backend-option
 * failure and then compiles successfully to verify that caller-provided consumer configuration is honored.
 */
TEST_P(CurtcFrontend, UsesExplicitConsumerSourceHeadersAndOptions)
{
    bool module = std::string_view(GetParam()) != "pch";
    std::string prefix = module ? "module;\n#include <clangrtc/device.hpp>\n"
                                  "export module curtc_consumer_prefix;\nexport "
                                : "";
    prefix.append("template<class T> __device__ T from_artifact(T value) { return value; }\n");
    auto flags = source().options;
    flags.insert(flags.end(), {std::format("--gpu-architecture={}", test::gpu_target()),
                               std::format("--artifact={}", GetParam()), "--emit=nvvm-ir-text"});
    auto prepared = test::prepare(test::request(prefix, {}), flags);
    ASSERT_EQ(prepared.status, CLANGRTC_SUCCESS) << prepared.log();
    auto package = prepared.output();
    auto environment = test::load(package);
    ASSERT_EQ(environment.status, CLANGRTC_SUCCESS);
    std::string consumer = module ? "import curtc_consumer_prefix;\n" : "";
    consumer.append("#include \"consumer_value.cuh\"\n"
                    "extern \"C\" __global__ void explicit_kernel(int* out) {\n"
                    "    *out = from_artifact(CONSUMER_VALUE);\n"
                    "}\n");
    clangrtcFile header{inputs::to_view("/consumer/consumer_value.cuh"),
                        inputs::to_view("#define CONSUMER_VALUE 42\n")};
    clangrtcString options[]{inputs::to_view("--nvvm-option=-curtc-invalid-consumer-option")};
    curtcInput input{};
    input.type = CURTC_INPUT_TYPE_ARTIFACT;
    input.source.artifact = environment.artifact.get();
    input.source.source = {inputs::to_view("/consumer/consumer.cu"), inputs::to_view(consumer)};
    input.source.headers = &header;
    input.source.headerCount = 1;
    input.options = options;
    input.optionCount = 1;
    curtcKernelSelection selected{inputs::to_view("&explicit_kernel"), {}};
    auto r = request(input);
    input.entries = &selected;
    test::TemporaryDirectory directory;
    auto owner = cache(directory);
    diagnostic log;
    curtcLibrary output{};
    EXPECT_EQ(curtcGet(owner.get(), &r, diagnostic::collect, &log, &output, nullptr), CLANGRTC_COMPILATION_FAILED);
    EXPECT_EQ(output, nullptr);
    EXPECT_NE(log.message.find("curtc-invalid-consumer-option"), std::string::npos) << log.message;

    options[0] = inputs::to_view("--nvvm-option=-ftz=1");
    auto library = get_library(owner.get(), r);
    EXPECT_NE(kernel_handle(library.get(), 0), nullptr);
    auto metadata = kernel_info(library.get(), 0);
    EXPECT_EQ(test::to_string(metadata.expression), "&explicit_kernel");
    EXPECT_EQ(test::to_string(metadata.symbolName), "explicit_kernel");
    EXPECT_EQ(test::to_string(metadata.cudaName), "explicit_kernel");
}

/** @brief Compile independent concurrent specializations from one immutable caller-owned artifact. */
TEST_P(CurtcFrontend, SharesReadyArtifactAcrossConcurrentSpecializations)
{
    bool module = std::string_view(GetParam()) != "pch";
    std::string prefix = module ? "module;\n#include <clangrtc/device.hpp>\n"
                                  "export module ready_program;\nexport "
                                : "";
    prefix.append("template<int N> __global__ void specialized(int* out) { *out = N; }\n");
    auto flags = source().options;
    flags.insert(flags.end(),
                 {std::format("--gpu-architecture={}", test::gpu_target()), std::format("--artifact={}", GetParam())});
    auto prepared = test::prepare(test::request(prefix, {}), flags);
    ASSERT_EQ(prepared.status, CLANGRTC_SUCCESS) << prepared.log();
    auto environment = test::load(prepared.output());
    ASSERT_EQ(environment.status, CLANGRTC_SUCCESS);
    prepared.result.reset();
    test::TemporaryDirectory directory;
    auto owner = cache(directory);
    std::vector<std::future<Library>> requests;

    for (int value = 40; value < 44; ++value)
    {
        requests.push_back(std::async(std::launch::async, [&, value] {
            auto expression = std::format("&specialized<{}>", value);
            curtcKernelSelection entry{inputs::to_view(expression), {}};
            curtcInput input{};
            input.type = CURTC_INPUT_TYPE_ARTIFACT;
            input.source.artifact = environment.artifact.get();
            auto r = request(input);
            input.entries = &entry;
            return get_library(owner.get(), r);
        }));
    }

    for (size_t i = 0; i < requests.size(); ++i)
    {
        auto library = requests[i].get();
        test::launch(kernel_handle(library.get(), 0), 40 + static_cast<int>(i));
    }

    EXPECT_EQ(statistics(owner.get()).kernelCompilations, requests.size());
}

/** @brief Run runtime artifact-consumer tests for PCH, BMI, and reduced-BMI formats. */
INSTANTIATE_TEST_SUITE_P(Formats, CurtcFrontend, ::testing::Values("pch", "bmi", "reduced-bmi"));

/**
 * @brief Reject an invalid CUDA image and permit a later request with a valid cubin.
 *
 * Supplies an invalid image to the actual driver, checks its diagnostic, and then loads and executes a valid cubin.
 */
TEST(Curtc, RejectsInvalidCudaImageAndCanRetry)
{
    test::TemporaryDirectory directory;
    auto owner = cache(directory);
    auto image = test::cubin_fixture();
    ASSERT_GE(image.size(), sizeof(Elf64_Ehdr));
    auto invalid_image = image;
    // Preserve the real cubin's layout while removing its CUDA machine identity.
    constexpr Elf64_Half invalid_machine = EM_NONE;
    std::memcpy(invalid_image.data() + offsetof(Elf64_Ehdr, e_machine), &invalid_machine, sizeof(invalid_machine));
    curtcInput input{};
    input.type = CURTC_INPUT_TYPE_CUBIN;
    input.data = {invalid_image.data(), invalid_image.size()};
    auto r = request(input);
    diagnostic log;
    curtcLibrary output{};
    EXPECT_EQ(curtcGet(owner.get(), &r, diagnostic::collect, &log, &output, nullptr), CLANGRTC_TOOLCHAIN_ERROR);
    EXPECT_EQ(output, nullptr);
    EXPECT_EQ(log.stage, "cuda");
    // Driver versions may reject the image during loading, enumeration, or symbol validation.
    EXPECT_FALSE(log.message.empty());
    input.data = {image.data(), image.size()};
    auto library = get_library(owner.get(), r);
    test::launch(kernel_handle(library.get(), 0), 42);
}

/**
 * @brief Keep a caller-owned library usable while cache retention is disabled.
 *
 * Disables memory and disk retention and executes the kernel while its final reference is live and releases it without
 * reusing a stale handle.
 */
TEST(Curtc, DisablingRetentionKeepsLibraryUsableUntilLastCaller)
{
    curtcKernelSelection selected{};
    curtcConfiguration config{};
    ASSERT_EQ(curtcDefaultConfiguration(&config), CLANGRTC_SUCCESS);
    config.memoryCache = false;
    config.diskCache = false;
    curtcCache output{};
    ASSERT_EQ(curtcCreateCache(&config, &output), CLANGRTC_SUCCESS);
    Cache owner(output, curtcDestroyCache);
    auto image = test::cubin_fixture();
    curtcInput input{};
    input.type = CURTC_INPUT_TYPE_CUBIN;
    input.data = {image.data(), image.size()};
    auto r = request(input);
    selected.name = inputs::to_view("fixed");
    input.entries = &selected;
    input.entryCount = 1;
    auto loaded = get_library(owner.get(), r);
    auto raw = kernel_handle(loaded.get(), 0);
    test::launch(raw, 42);
    loaded.reset();
}

/**
 * @brief Reject invalid requests while preserving compiler-specific error status and safe outputs.
 *
 * Covers NULL arguments, missing inputs, invalid kinds and selectors, statistics calls, and exceptions thrown by a
 * diagnostic callback.
 */
TEST(Curtc, InvalidArgumentsClearOutputsAndDiagnosticsPreserveCompilerStatus)
{
    curtcKernelSelection selected{};
    test::TemporaryDirectory directory;
    auto owner = cache(directory);
    auto result = reinterpret_cast<curtcLibrary>(1);
    EXPECT_EQ(curtcGet(owner.get(), nullptr, nullptr, nullptr, &result, nullptr), CLANGRTC_INVALID_ARGUMENT);
    EXPECT_EQ(result, nullptr);
    curtcRequest r{};
    EXPECT_EQ(curtcGet(nullptr, &r, nullptr, nullptr, &result, nullptr), CLANGRTC_INVALID_ARGUMENT);
    EXPECT_EQ(curtcGet(owner.get(), &r, nullptr, nullptr, &result, nullptr), CLANGRTC_INVALID_ARGUMENT);
    EXPECT_EQ(result, nullptr);
    selected.name = inputs::to_view("fixed");
    curtcInput invalid{};
    invalid.entries = &selected;
    invalid.entryCount = 1;
    invalid.type = static_cast<curtcInputType>(99);
    r.inputs = &invalid;
    r.inputCount = 1;
    diagnostic argument_log;
    EXPECT_EQ(curtcGet(owner.get(), &r, diagnostic::collect, &argument_log, &result, nullptr),
              CLANGRTC_INVALID_ARGUMENT);
    EXPECT_EQ(argument_log.stage, "request");
    EXPECT_NE(argument_log.message.find("curtc.cpp:"), std::string::npos);
    EXPECT_NE(argument_log.message.find("Unsupported curtc input type: 99"), std::string::npos);
    curtcCacheStatistics stats{};
    EXPECT_EQ(curtcGetStatistics(nullptr, &stats), CLANGRTC_INVALID_ARGUMENT);
    EXPECT_EQ(curtcGetStatistics(owner.get(), nullptr), CLANGRTC_INVALID_ARGUMENT);
    EXPECT_EQ(curtcDefaultConfiguration(nullptr), CLANGRTC_INVALID_ARGUMENT);
    EXPECT_EQ(curtcCreateCache(nullptr, nullptr), CLANGRTC_INVALID_ARGUMENT);

    auto input = source();
    inputs::source_view view(input);
    r = request(view.value);
    selected.name = inputs::to_view("&missing_kernel");
    view.value.entries = &selected;
    view.value.entryCount = 1;
    diagnostic log;
    EXPECT_EQ(curtcGet(owner.get(), &r, diagnostic::collect, &log, &result, nullptr), CLANGRTC_INVALID_EXPRESSION);
    EXPECT_EQ(result, nullptr);
    EXPECT_NE(log.message.find("missing_kernel"), std::string::npos);
    r = request(view.value);
    selected.name = {};
    view.value.entries = &selected;
    auto throws = [](clangrtcDiagnostic const *, void *) { throw std::runtime_error("callback"); };
    EXPECT_EQ(curtcGet(owner.get(), &r, throws, nullptr, &result, nullptr), CLANGRTC_INVALID_ARGUMENT);
}
} // namespace

/**
 * @brief Share successful explicit runtime initialization across concurrent and repeated callers.
 *
 * Runs overlapping initialization calls and verifies that subsequent initialization still succeeds.
 */
TEST(Curtc, ExplicitInitializationIsConcurrentAndIdempotent)
{
    std::vector<std::future<clangrtcStatus>> requests;

    for (int i = 0; i < 8; ++i)
    {
        requests.push_back(std::async(std::launch::async, [] { return curtcInit(nullptr, nullptr); }));
    }

    for (auto &request : requests)
    {
        EXPECT_EQ(request.get(), CLANGRTC_SUCCESS);
    }

    EXPECT_EQ(curtcInit(nullptr, nullptr), CLANGRTC_SUCCESS);
}
