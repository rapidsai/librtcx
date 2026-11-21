/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* SPDX-License-Identifier: Apache-2.0 */

/**
 * @file package_test.cpp
 * @brief Test artifact validation beyond the package integrity envelope.
 *
 * Recomputes package digests after deliberate metadata or bitcode edits so failures exercise inner validation and
 * diagnostic recovery.
 */

#include "artifact_package.hpp"
#include "support.hpp"

#include <format>
#include <llvm/ADT/StringExtras.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/MemoryBuffer.h>
#include <stdexcept>

namespace test
{

/** @brief Match canonical upstream XXH3-128 vectors, including an empty byte range. */
TEST(ContentHash, UsesCanonicalXxh3_128Identities)
{
    EXPECT_EQ(cr::content_hash({}), "99aa06d3014798d86001c324468d497f");
    EXPECT_EQ(cr::content_hash(std::span<char const>("abc", 3)), "06b05ab6733a618578af5f94892f3950");
}

namespace
{
// Recompute integrity fields so these tests exercise inner package validation.
template <class Mutate> std::string rewrite_package(std::string const &package, Mutate mutate)
{
    auto first = package.find('\n'), second = package.find('\n', first + 1), third = package.find('\n', second + 1);
    auto size = std::stoull(package.substr(first + 1, second - first - 1));
    auto parsed = llvm::json::parse(llvm::StringRef(package.data() + third + 1, size));

    if (!parsed || !parsed->getAsObject())
    {
        throw std::runtime_error("Invalid fixture manifest");
    }

    auto &meta = *parsed->getAsObject();
    auto body = package.substr(third + 1 + size);
    mutate(meta, body);
    std::string json;
    llvm::raw_string_ostream output(json);
    output << *parsed;
    auto bytes = json;
    bytes.append(body);
    auto digest = cr::content_hash(bytes);
    return std::format("{}\n{}\n{}\n{}", cr::artifact_package_magic, json.size(), digest, bytes);
}

} // namespace

class PackageTest : public CompilationTest
{
  protected:
    std::string package;

    void SetUp() override
    {
        CompilationTest::SetUp();
        options.push_back("--artifact=cuda");
        auto slot = to_view("&prepared");
        auto result = prepare(request("extern \"C\" __device__ int prepared() { return 7; }", {&slot, 1}), options);
        ASSERT_EQ(result.status, CLANGRTC_SUCCESS) << result.log();
        package = result.output();
    }
};

/** @brief Read the existing stored digest without copying the package and keep full load-time verification. */
TEST_F(PackageTest, ExposesStoredContentIdentityAndValidatesItWhenLoading)
{
    auto envelope = cr::read_artifact_envelope({package.data(), package.size()});
    EXPECT_EQ(envelope.checksum.size(), cr::artifact_checksum_size);
    EXPECT_EQ(envelope.contents.data() + envelope.contents.size(), package.data() + package.size());
    auto expected = cr::content_hash(envelope.contents);
    EXPECT_EQ(envelope.checksum, expected);
    auto loaded = load(package);
    ASSERT_EQ(loaded.status, CLANGRTC_SUCCESS);
    clangrtcArtifactInfo info{};
    ASSERT_EQ(clangrtcGetArtifactInfo(loaded.artifact.get(), &info), CLANGRTC_SUCCESS);
    EXPECT_EQ(to_string(info.contentHash), expected);

    auto corrupt = package;
    corrupt.back() ^= 1;
    auto identity = cr::read_artifact_envelope({corrupt.data(), corrupt.size()});
    EXPECT_EQ(identity.checksum, envelope.checksum);
    auto rejected = load(corrupt);
    EXPECT_EQ(rejected.status, CLANGRTC_INVALID_ARTIFACT);
    EXPECT_EQ(rejected.artifact.get(), nullptr);
}

/** @brief Reject malformed checksum framing before considering an artifact as a cache identity. */
TEST_F(PackageTest, RejectsMalformedStoredDigest)
{
    auto envelope = cr::read_artifact_envelope({package.data(), package.size()});
    auto offset = static_cast<size_t>(envelope.checksum.data() - package.data());
    auto malformed = package;
    malformed[offset] = 'g';
    auto rejected = load(malformed);
    EXPECT_EQ(rejected.status, CLANGRTC_INVALID_ARTIFACT);
    EXPECT_EQ(rejected.artifact.get(), nullptr);
}

/**
 * @brief Reject invalid artifact metadata even when the package integrity fields are correct.
 *
 * Removes a required metadata field and recomputes the package envelope so inner validation must detect the error.
 */
TEST_F(PackageTest, RejectsMalformedMetadata)
{
    auto malformed = rewrite_package(package, [](auto &meta, auto &) { meta.erase("target"); });
    auto result = load(malformed);
    EXPECT_EQ(result.status, CLANGRTC_INVALID_ARTIFACT);
    EXPECT_EQ(result.artifact.get(), nullptr);
}

/**
 * @brief Distinguish an incompatible producer identity from corrupt package data.
 *
 * Changes the recorded compiler identity while preserving package integrity and checks the incompatible-artifact
 * status.
 */
TEST_F(PackageTest, DistinguishesCompilerMismatchFromCorruption)
{
    auto incompatible = rewrite_package(package, [](auto &meta, auto &) { meta["compiler"] = "another build"; });
    auto result = load(incompatible);
    EXPECT_EQ(result.status, CLANGRTC_INCOMPATIBLE_ARTIFACT);
    EXPECT_EQ(result.artifact.get(), nullptr);
}

/**
 * @brief Report invalid prepared bitcode when an otherwise valid artifact is consumed.
 *
 * Corrupts the embedded IR with a valid outer digest and requires compilation-time artifact rejection without output.
 */
TEST_F(PackageTest, ReportsMalformedPreparedBitcode)
{
    auto invalid = rewrite_package(package, [](auto &meta, auto &body) {
        auto offset = *meta.getInteger("payload") + *meta.getInteger("bootstrap");
        body.at(static_cast<size_t>(offset)) ^= 1;
    });
    auto loaded = load(invalid);
    ASSERT_EQ(loaded.status, CLANGRTC_SUCCESS);
    auto input = request("extern \"C\" __global__ void consumer() {}", {});
    input.source.path = to_view("/__clangrtc__/consumer.cu");
    input.artifact = loaded.artifact.get();
    auto result = compile(input, {});
    EXPECT_EQ(result.status, CLANGRTC_INVALID_ARTIFACT);
    EXPECT_TRUE(result.output().empty());
    EXPECT_FALSE(result.log().empty());
}

/**
 * @brief Recover from an LLVM IR link failure while preserving the linker diagnostic.
 *
 * Edits prepared module flags to trigger a conflict and checks that subsequent compilation remains usable.
 */
TEST_F(PackageTest, LinkFailureIsRecoverableAndPreservesLlvmDiagnostic)
{
    auto conflicting = rewrite_package(package, [](auto &meta, auto &body) {
        auto offset = static_cast<size_t>(*meta.getInteger("payload") + *meta.getInteger("bootstrap"));
        auto length = static_cast<size_t>(*meta.getInteger("ir"));
        llvm::LLVMContext context;
        auto module = llvm::parseBitcodeFile(
            llvm::MemoryBufferRef(llvm::StringRef(body.data() + offset, length), "fixture"), context);

        if (!module)
        {
            throw std::runtime_error(llvm::toString(module.takeError()));
        }

        bool changed = false;

        for (auto *flag : (*module)->getNamedMetadata("llvm.module.flags")->operands())
        {
            if (llvm::cast<llvm::MDString>(flag->getOperand(1))->getString() == "frame-pointer")
            {
                flag->replaceOperandWith(0, llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                                                llvm::Type::getInt32Ty(context), llvm::Module::Error)));
                changed = true;
            }
        }

        if (!changed)
        {
            throw std::runtime_error("Missing fixture frame-pointer flag");
        }

        std::string ir;
        llvm::raw_string_ostream output(ir);
        llvm::WriteBitcodeToFile(**module, output);
        body.replace(offset, length, ir);
        meta["ir"] = ir.size();
    });
    auto conflict = load(conflicting);
    ASSERT_EQ(conflict.status, CLANGRTC_SUCCESS);
    auto input = request("extern \"C\" __global__ void consumer() {}", {});
    input.source.path = to_view("/__clangrtc__/consumer.cu");
    input.artifact = conflict.artifact.get();
    auto result = compile(input, {});
    EXPECT_EQ(result.status, CLANGRTC_LINK_FAILED);
    EXPECT_TRUE(result.output().empty());
    EXPECT_NE(result.log().find("frame-pointer"), std::string::npos);

    // A failed LLVM link must restore its diagnostic handler and leave later
    // compilations usable in the same process.
    auto valid = load(package);
    ASSERT_EQ(valid.status, CLANGRTC_SUCCESS);
    input.artifact = valid.artifact.get();
    auto recovered = compile(input, {});
    EXPECT_EQ(recovered.status, CLANGRTC_SUCCESS) << recovered.log();
}

/**
 * @brief Report errors in appended consumer source using the consumer filename and line.
 *
 * Loads a valid package, adds an error directive, and checks that diagnostics do not incorrectly attribute it to the
 * prepared source.
 */
TEST_F(PackageTest, ConsumerDiagnosticKeepsConsumerLocation)
{
    auto loaded = load(package);
    ASSERT_EQ(loaded.status, CLANGRTC_SUCCESS);
    auto input = request("#error consumer source failure\n", {});
    input.source.path = to_view("/__clangrtc__/consumer.cu");
    input.source.path = to_view("/__clangrtc__/consumer.cu");
    input.artifact = loaded.artifact.get();
    auto result = compile(input, {});
    EXPECT_EQ(result.status, CLANGRTC_COMPILATION_FAILED);
    EXPECT_NE(result.log().find("/__clangrtc__/consumer.cu:1:"), std::string::npos);
}
} // namespace test
