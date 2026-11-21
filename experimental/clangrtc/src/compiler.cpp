/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file compiler.cpp
 * @brief Implements compiler requests, in-memory artifacts, and the public C API.
 */
#include "artifact_package.hpp"
#include "clangrtc.h"
#include "embedded.hpp"
#include "error.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <clang/AST/Attr.h>
#include <clang/AST/Decl.h>
#include <clang/AST/Expr.h>
#include <clang/AST/Mangle.h>
#include <clang/Basic/Module.h>
#include <clang/Basic/Version.h>
#include <clang/CodeGen/CodeGenAction.h>
#include <clang/Driver/Compilation.h>
#include <clang/Driver/Driver.h>
#include <clang/Driver/Job.h>
#include <clang/Driver/Tool.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/CompilerInvocation.h>
#include <clang/Frontend/FrontendActions.h>
#include <clang/Frontend/MultiplexConsumer.h>
#include <clang/Lex/Lexer.h>
#include <clang/Lex/Preprocessor.h>
#include <clang/Lex/PreprocessorOptions.h>
#include <clang/Options/Options.h>
#include <clang/Sema/Sema.h>
#include <clang/Serialization/ASTWriter.h>
#include <clang/Serialization/ModuleCache.h>
#include <cstdlib>
#include <dlfcn.h>
#include <exception>
#include <filesystem>
#include <format>
#include <fstream>
#include <llvm/ADT/IntrusiveRefCntPtr.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/Config/llvm-config.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DiagnosticHandler.h>
#include <llvm/IR/DiagnosticInfo.h>
#include <llvm/IR/DiagnosticPrinter.h>
#include <llvm/IR/GlobalObject.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Linker/Linker.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Option/ArgList.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/TimeProfiler.h>
#include <llvm/Support/VirtualFileSystem.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/TargetParser/Host.h>
#include <llvm/Transforms/IPO/Internalize.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <nvvm.h>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

/**
 * @brief Checks an NVVM call, preserving its expression, call site, and program log.
 *
 * @param call NVVM expression evaluated exactly once.
 * @param program NVVM program used to retrieve diagnostics; may be null.
 * @throws cr::Failure with compilation-failed status and the nvvm diagnostic stage.
 */
#define CR_CHECK_NVVM(call, program)                                                                                   \
    do                                                                                                                 \
    {                                                                                                                  \
        auto const cr_nvvm_status = (call);                                                                            \
                                                                                                                       \
        if (cr_nvvm_status != NVVM_SUCCESS)                                                                            \
        {                                                                                                              \
            auto const cr_nvvm_program = (program);                                                                    \
            size_t cr_nvvm_size{};                                                                                     \
            std::string cr_nvvm_log;                                                                                   \
                                                                                                                       \
            if (cr_nvvm_program &&                                                                                     \
                ::cr::nvvm().nvvmGetProgramLogSize(cr_nvvm_program, &cr_nvvm_size) == NVVM_SUCCESS && cr_nvvm_size)    \
            {                                                                                                          \
                cr_nvvm_log.resize(cr_nvvm_size);                                                                      \
                                                                                                                       \
                if (::cr::nvvm().nvvmGetProgramLog(cr_nvvm_program, cr_nvvm_log.data()) != NVVM_SUCCESS)               \
                {                                                                                                      \
                    cr_nvvm_log.clear();                                                                               \
                }                                                                                                      \
                                                                                                                       \
                if (!cr_nvvm_log.empty() && cr_nvvm_log.back() == '\0')                                                \
                {                                                                                                      \
                    cr_nvvm_log.pop_back();                                                                            \
                }                                                                                                      \
            }                                                                                                          \
                                                                                                                       \
            throw ::cr::Failure(CLANGRTC_COMPILATION_FAILED,                                                           \
                                std::format("clangrtc failure at {}:{}: libNVVM call `{}` failed: {}\n{}", __FILE__,   \
                                            __LINE__, #call, ::cr::nvvm().nvvmGetErrorString(cr_nvvm_status),          \
                                            cr_nvvm_log),                                                              \
                                "nvvm");                                                                               \
        }                                                                                                              \
    } while (false)

/**
 * @brief Apply a macro to every required libNVVM entry point.
 * @param entry Macro accepting the full external entry-point name.
 */
// clang-format off
#define CR_NVVM_ENTRIES(entry)                \
    entry(nvvmVersion)                        \
    entry(nvvmCreateProgram)                  \
    entry(nvvmDestroyProgram)                 \
    entry(nvvmAddModuleToProgram)             \
    entry(nvvmCompileProgram)                 \
    entry(nvvmGetCompiledResultSize)          \
    entry(nvvmGetCompiledResult)              \
    entry(nvvmGetProgramLogSize)              \
    entry(nvvmGetProgramLog)                  \
    entry(nvvmGetErrorString)
// clang-format on

/**
 * @brief Declare a libNVVM function pointer with its external name and ABI.
 * @param name Full external entry-point name.
 */
#define CR_NVVM_ENTRY(name) decltype(&::name) name{};

/**
 * @brief Bind a declared libNVVM function pointer according to the configured linkage.
 * @param name Full external entry-point name.
 */
#if CLANGRTC_STATIC_LINK_NVVM
#define CR_LOAD_NVVM_ENTRY(name) name = &::name;
#else
#define CR_LOAD_NVVM_ENTRY(name) name = symbol<decltype(name)>(#name);
#endif

/**
 * @brief Implementation helpers for compilation and artifact storage.
 *
 * Helpers may throw Failure or allocation exceptions. Public API entry points
 * translate supported failures into clangrtcStatus values.
 */
namespace cr
{
/**
 * @brief Library version supplied by the CMake project definition.
 */
constexpr std::string_view libraryVersion = CLANGRTC_VERSION;

/** @brief Prepared-state ABI, including the native CUDA header and compiler-identity contract. */
constexpr std::string_view artifact_compiler_abi = "clangrtc-package-2";

/** @brief Own the libNVVM loader handle and its immutable C API dispatch table. */
class nvvm_context
{
#if !CLANGRTC_STATIC_LINK_NVVM
    /** @brief Owned shared-library handle, kept open while the dispatch table is in use. */
    void *library_{};

    /**
     * @brief Resolve a required libNVVM entry point from the owned shared library.
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
    /** @brief Resolved nvvmVersion entry point. */
    CR_NVVM_ENTRY(nvvmVersion)

    /** @brief Resolved nvvmCreateProgram entry point. */
    CR_NVVM_ENTRY(nvvmCreateProgram)

    /** @brief Resolved nvvmDestroyProgram entry point. */
    CR_NVVM_ENTRY(nvvmDestroyProgram)

    /** @brief Resolved nvvmAddModuleToProgram entry point. */
    CR_NVVM_ENTRY(nvvmAddModuleToProgram)

    /** @brief Resolved nvvmCompileProgram entry point. */
    CR_NVVM_ENTRY(nvvmCompileProgram)

    /** @brief Resolved nvvmGetCompiledResultSize entry point. */
    CR_NVVM_ENTRY(nvvmGetCompiledResultSize)

    /** @brief Resolved nvvmGetCompiledResult entry point. */
    CR_NVVM_ENTRY(nvvmGetCompiledResult)

    /** @brief Resolved nvvmGetProgramLogSize entry point. */
    CR_NVVM_ENTRY(nvvmGetProgramLogSize)

    /** @brief Resolved nvvmGetProgramLog entry point. */
    CR_NVVM_ENTRY(nvvmGetProgramLog)

    /** @brief Resolved nvvmGetErrorString entry point. */
    CR_NVVM_ENTRY(nvvmGetErrorString)

    /**
     * @brief Initialize libNVVM's dispatch table using the selected static or shared backend.
     * @throws Failure With toolchain-error status if loading or symbol resolution fails.
     * @note Constructor failures close the shared-library handle before propagating.
     */
    nvvm_context()
    {
#if !CLANGRTC_STATIC_LINK_NVVM
        constexpr char const *library_path = CLANGRTC_NVVM_LIBRARY;
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
            CR_NVVM_ENTRIES(CR_LOAD_NVVM_ENTRY)
        }
        catch (...)
        {
#if !CLANGRTC_STATIC_LINK_NVVM
            ::dlclose(library_);
#endif

            throw;
        }
    }

    /** @brief Disable copying of the backend context and its owned handle. */
    nvvm_context(nvvm_context const &) = delete;

    /** @brief Disable assignment of the backend context and its owned handle. */
    nvvm_context &operator=(nvvm_context const &) = delete;

    /** @brief Close the owned libNVVM shared library when using dynamic linking. */
    ~nvvm_context()
    {
#if !CLANGRTC_STATIC_LINK_NVVM
        ::dlclose(library_);
#endif
    }
};

/** @brief Lazily initialize libNVVM once; a failed initialization may be retried. */
nvvm_context const &nvvm()
{
    static nvvm_context const context;
    return context;
}

/**
 * @brief Exclusive entry-count limit for request arrays and artifact metadata lists.
 */
constexpr size_t entry_count_limit = 100'000;

/**
 * @brief Generated alias prefix for slots stored in a prepared artifact.
 */
constexpr std::string_view artifact_expression_prefix = "__clangrtc_artifact_expression_";

/**
 * @brief Generated alias prefix for slots belonging to one compilation request.
 */
constexpr std::string_view compilation_expression_prefix = "__clangrtc_expression_";

/**
 * @brief Presumed source directory identifying diagnostics from requested expression slots.
 */
constexpr std::string_view expression_source_prefix = "/__clangrtc__/expressions/";

/**
 * @brief Buffers one compiler output and publishes it only when kept.
 *
 * Each file owns its pending bytes. The shared output registry outlives any
 * backend clone that created the file. Keeping the file replaces the registry
 * entry for its virtual path; discarding it leaves the registry unchanged.
 */
struct MemoryFile : llvm::vfs::OutputFileImpl
{
    /**
     * @brief Shared registry of successfully kept outputs, keyed by virtual path.
     */
    std::shared_ptr<std::map<std::string, std::string>> files;

    /**
     * @brief Virtual path under which keep() publishes the buffered bytes.
     */
    std::string path;

    /**
     * @brief Pending output owned by this file until it is kept or discarded.
     */
    llvm::SmallString<0> bytes;

    /**
     * @brief Random-access output stream backed by bytes.
     */
    llvm::raw_svector_ostream stream{bytes};

    /**
     * @brief Opens an initially empty output without modifying the registry.
     *
     * @param files Shared registry receiving the output if it is kept.
     * @param path Virtual output path copied into this file.
     */
    MemoryFile(std::shared_ptr<std::map<std::string, std::string>> files, std::string path)
        : files(std::move(files)), path(std::move(path))
    {
    }

    /**
     * @brief Copies buffered bytes into the registry, replacing any previous entry.
     *
     * @return LLVM success after publishing the output.
     */
    llvm::Error keep() override
    {
        // Publish only completed outputs; a failed frontend can discard its partial buffer.
        (*files)[path] = bytes.str().str();
        return llvm::Error::success();
    }

    /**
     * @brief Abandons this write without changing previously published output.
     *
     * @return LLVM success; no filesystem cleanup is needed.
     */
    llvm::Error discard() override
    {
        return llvm::Error::success();
    }

    /**
     * @brief Provides the stream used by Clang to serialize this output.
     *
     * @return Stream borrowed until this file is destroyed.
     */
    llvm::raw_pwrite_stream &getOS() override
    {
        return stream;
    }
};

/**
 * @brief Captures CompilerInstance outputs in memory instead of writing files.
 *
 * Backend clones share one registry so their kept outputs remain visible to
 * captured(). This registry belongs to one compilation; it does not synchronize
 * concurrent writers. Filesystem-specific output settings have no effect.
 */
struct MemoryOutputs : llvm::vfs::OutputBackend
{
    /**
     * @brief Output registry shared by backend clones and their open files.
     */
    std::shared_ptr<std::map<std::string, std::string>> files = std::make_shared<std::map<std::string, std::string>>();

    /**
     * @brief Creates a backend sharing the registry rather than copying its data.
     *
     * @return New reference-counted backend attached to the same registry.
     */
    llvm::IntrusiveRefCntPtr<llvm::vfs::OutputBackend> cloneImpl() const override
    {
        auto clone = llvm::makeIntrusiveRefCnt<MemoryOutputs>();
        clone->files = files;
        return clone;
    }

    /**
     * @brief Creates a buffered output without publishing or touching the filesystem.
     *
     * @param path Virtual path copied into the new file.
     * @param config Filesystem output settings, unused by this memory backend.
     * @return Owned file that publishes its bytes only when kept.
     */
    llvm::Expected<std::unique_ptr<llvm::vfs::OutputFileImpl>> createFileImpl(
        llvm::StringRef path, std::optional<llvm::vfs::OutputConfig> config) override
    {
        return std::unique_ptr<llvm::vfs::OutputFileImpl>(new MemoryFile(files, path.str()));
    }
};

/**
 * @brief Copies the primary output captured by this compiler instance.
 *
 * The output backend must be MemoryOutputs. Looking up a missing primary
 * output creates an empty registry entry.
 *
 * @param ci Compiler instance whose primary virtual output is requested.
 * @return Owned bytes recorded at /__clangrtc__/output, or an empty string.
 */
std::string captured(clang::CompilerInstance &ci)
{
    return (*static_cast<MemoryOutputs &>(ci.getOutputManager()).files)["/__clangrtc__/output"];
}

/**
 * @brief Serializes a full or reduced named-module BMI into caller-owned storage.
 *
 * Clang's CXX20ModulesGenerator writes directly to a raw_fd_ostream, bypassing
 * the CompilerInstance output backend. This consumer uses the same public AST
 * writer but copies the completed buffer into memory instead. Both referenced
 * strings must outlive this consumer and its translation-unit callback.
 */
struct MemoryModuleGenerator : clang::CXX20ModulesGenerator
{
    /** @brief Compiler owning the semantic state consulted during serialization. */
    clang::CompilerInstance &compiler;

    /** @brief Whether this consumer writes a reduced module interface. */
    bool reduced;

    /**
     * @brief Destination for the completed serialized module bytes.
     */
    std::string &output;

    /**
     * @brief Destination for the named module's identity before AST teardown.
     */
    std::string &moduleName;

    /**
     * @brief Configures Clang's module writer and borrows the result destinations.
     *
     * @param ci Compiler instance owning the preprocessor and module cache.
     * @param reduced Whether to omit declarations unnecessary for a reduced BMI.
     * @param output Destination for serialized bytes after successful generation.
     * @param moduleName Destination for the module name obtained from the AST.
     */
    MemoryModuleGenerator(clang::CompilerInstance &ci, bool reduced, std::string &output, std::string &moduleName)
        : CXX20ModulesGenerator(ci.getPreprocessor(), ci.getModuleCache(), ci.getFrontendOpts().OutputFile,
                                ci.getCodeGenOpts(), reduced, false),
          compiler(ci), reduced(reduced), output(output), moduleName(moduleName)
    {
    }

    /**
     * @brief Records the module name and captures the completed serialized AST.
     *
     * Calls PCHGenerator directly to avoid the base module generator's disk
     * write. The output is changed only if serialization completed successfully.
     *
     * @param context AST containing the named module being prepared.
     */
    void HandleTranslationUnit(clang::ASTContext &context) override
    {
        if (auto *module = context.getCurrentNamedModule())
        {
            moduleName = module->Name;
        }

        auto &sema = compiler.getSema();
        // Clang otherwise makes every CUDA diagnostic candidate a serialization root, even in a
        // reduced BMI. Completed analysis with no deferred diagnostics needs no extra roots;
        // ordinary module reachability still retains declarations used by exported templates.
        auto prune = reduced && sema.getLangOpts().CUDA && !sema.getLangOpts().OpenMP && !sema.getLangOpts().OpenACC &&
                     sema.DeviceDeferredDiags.empty() && !compiler.getDiagnostics().hasErrorOccurred();
        llvm::SmallSetVector<clang::Decl *, 4> diagnosticRoots;

        if (prune)
        {
            diagnosticRoots.swap(sema.DeclsToCheckForDeferredDiags);
        }

        CR_DEFER([&] {
            if (prune)
            {
                diagnosticRoots.swap(sema.DeclsToCheckForDeferredDiags);
            }
        });

        clang::PCHGenerator::HandleTranslationUnit(context);

        if (isComplete())
        {
            output.assign(getPCH().data(), getPCH().size());
        }
    }
};

/**
 * @brief Runs Clang's module-interface frontend with an in-memory BMI consumer.
 *
 * Retains the standard module-interface setup while replacing its AST consumer
 * with MemoryModuleGenerator. The caller owns the result strings and must keep
 * them alive until the action and its consumer have been destroyed.
 */
struct MemoryModuleAction : clang::GenerateModuleInterfaceAction
{
    /**
     * @brief Whether the consumer should generate a reduced BMI.
     */
    bool reduced;

    /**
     * @brief Borrowed destination for the serialized module bytes.
     */
    std::string &output;

    /**
     * @brief Borrowed destination for the named module's identity.
     */
    std::string &moduleName;

    /**
     * @brief Selects the BMI format and borrows the result destinations.
     *
     * @param reduced Whether to generate a reduced rather than full BMI.
     * @param output Destination for the generated BMI bytes.
     * @param moduleName Destination for the module name.
     */
    MemoryModuleAction(bool reduced, std::string &output, std::string &moduleName)
        : reduced(reduced), output(output), moduleName(moduleName)
    {
    }

    /**
     * @brief Creates the consumer that captures this action's module output.
     *
     * @param ci Compiler instance whose frontend state is used by the consumer.
     * @param sourcePath Input filename; the consumer obtains it from compiler state.
     * @return Owned consumer borrowing this action's output destinations.
     */
    std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(clang::CompilerInstance &ci,
                                                          llvm::StringRef sourcePath) override
    {
        return std::make_unique<MemoryModuleGenerator>(ci, reduced, output, moduleName);
    }
};

/**
 * @brief Monotonic clock used for operation and stage durations.
 */
using Clock = std::chrono::steady_clock;

/**
 * @brief Dependency paths mapped to modification times and owned file contents.
 *
 * Each pair contains the modification time in seconds since the epoch,
 * followed by the file bytes. Aliases may retain the same file under several paths.
 */
using Files = std::map<std::string, std::pair<int64_t, std::string>>;

/**
 * @brief Measures host elapsed time from a monotonic timestamp.
 *
 * @param start Beginning of the measured interval.
 * @return Elapsed nanoseconds at the time of this call.
 */
uint64_t elapsed(Clock::time_point start)
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();
}

/**
 * @brief Borrows contiguous text without reading a terminator.
 *
 * @param s Text whose underlying storage must outlive the returned view.
 * @return Public pointer/length pair borrowing the same storage.
 */
clangrtcString to_view(std::string_view s)
{
    return {s.data(), s.size()};
}

/**
 * @brief Borrows NUL-terminated text as a length-delimited view.
 *
 * @param s Non-NULL text whose terminating NUL can be read.
 * @return Borrowed view excluding the terminating NUL byte.
 */
constexpr clangrtcString to_view(char const *s)
{
    return {s, std::char_traits<char>::length(s)};
}

/**
 * @brief Copies the specified bytes from a trusted public string view.
 *
 * No terminator is read and embedded NUL bytes are preserved. The caller
 * supplies a valid pointer/length pair; this helper does not validate it.
 *
 * @param value Borrowed input whose byte range is valid for the call.
 * @return Owned copy of exactly value.size bytes.
 */
std::string to_string(clangrtcString value)
{
    return {value.data, value.size};
}

/**
 * @brief Copies and validates an array of text views.
 *
 * @param p Borrowed array; may be NULL when n is zero.
 * @param n Entry count, which must be less than entry_count_limit.
 * @return Owned strings in input order.
 * @throws Failure if the array pointer or entry count is invalid.
 */
std::vector<std::string> to_strings(clangrtcString const *p, size_t n)
{
    CR_EXPECTS(p || !n, "NULL string array");
    CR_EXPECTS(n < entry_count_limit, "Too many options/expressions");
    std::vector<std::string> result;

    for (size_t i = 0; i < n; ++i)
    {
        result.push_back(to_string(p[i]));
    }

    return result;
}

/**
 * @brief Normalizes a virtual filename to an absolute lexical path.
 *
 * Relative names resolve against the process working directory. This
 * operation does not resolve symlinks or require the path to exist.
 *
 * @param name Requested filename, or empty to select the internal default.
 * @return Normalized path, using /__clangrtc__/source.cu for an empty name.
 */
std::string path(std::string name)
{
    if (name.empty())
    {
        return "/__clangrtc__/source.cu";
    }

    return std::filesystem::absolute(name).lexically_normal().string();
}

/**
 * @brief Reads a host file as binary bytes.
 *
 * @param name Filesystem path to open.
 * @return Owned file contents.
 * @throws Failure if the file cannot be opened.
 */
std::string read(std::string const &name)
{
    std::ifstream input(name, std::ios::binary);
    CR_EXPECTS(static_cast<bool>(input), std::format("Cannot read toolchain file {}", name), Failure,
               CLANGRTC_TOOLCHAIN_ERROR);
    return {std::istreambuf_iterator<char>(input), {}};
}

/**
 * @brief Identifies the prepared-state ABI and linked compiler build.
 *
 * @return Fingerprint used to reject packages prepared by another compiler build.
 */
std::string fingerprint()
{
    return std::format("{}/{}/{}", artifact_compiler_abi, clang::getClangFullVersion(), LLVM_VERSION_STRING);
}

/**
 * @brief Computes a canonical lowercase hexadecimal XXH3-128 digest.
 *
 * @param bytes Bytes to hash; embedded NULs are permitted.
 * @return XXH3-128 digest encoded as 32 hexadecimal characters.
 */
std::string hash(std::span<char const> bytes)
{
    return content_hash(bytes);
}

/**
 * @brief Holds validated option values and ordered frontend flags.
 *
 * Defaults select device-only compilation with the NVRTC header profile.
 * The option parser leaves caller-controlled macro definitions in flags.
 */
struct Settings
{
    /**
     * @brief Normalized CUDA GPU target, using the sm_NN spelling.
     */
    std::string target = "sm_75";

    /**
     * @brief Whether to compile CUDA device code instead of native CPU code.
     */
    bool device = true;

    /**
     * @brief Require requested entry expressions to resolve CUDA kernel definitions.
     */
    bool requireKernel = false;

    /** @brief Omit texture, surface, and CUDA runtime API declarations. */
    bool minimal = false;

    /** @brief Forward device standard-library headers to freestanding libcudacxx. */
    bool libcudacxx = false;

    /**
     * @brief Host source language (c or c++).
     */
    std::string language = "c++";

    /**
     * @brief CUDA header environment: nvrtc or hosted.
     */
    std::string profile = "nvrtc";

    /**
     * @brief Requested LLVM IR output format.
     */
    std::string format = "nvvm-ir-bitcode";

    /**
     * @brief Artifact format selected for explicit preparation.
     */
    std::string artifact = "reduced-bmi";

    /**
     * @brief CUDA toolkit root, with environment fallback when empty.
     */
    std::string cuda = CLANGRTC_CUDA_PATH;

    /**
     * @brief Clang resource directory supplying compiler builtin headers.
     */
    std::string resource = CLANGRTC_RESOURCE_DIR;

    /**
     * @brief Validated frontend flags, preserving preprocessing order.
     */
    std::vector<std::string> flags;

    /**
     * @brief Ordered backend arguments; excluded from frontend artifact identity.
     */
    std::vector<std::string> nvvmOptions;

    /**
     * @brief LLVM optimization level from zero through three.
     */
    int optimization = 2;

    /**
     * @brief Whether source is already preprocessed CUDA C++.
     */
    bool preprocessed = false;

    /**
     * @brief Whether the operation owns an LLVM time-trace profiling session.
     */
    bool trace = false;
};

/**
 * @brief Validates pass-through frontend arguments without allowing pipeline replacement.
 * @param flags Ordered Clang driver arguments.
 */
void validate_clang_options(std::vector<std::string> const &flags)
{
    std::vector<char const *> args;

    for (auto const &flag : flags)
    {
        args.push_back(flag.c_str());
    }

    unsigned missingIndex{}, missingCount{};
    auto parsed = clang::getDriverOptTable().ParseArgs(args, missingIndex, missingCount);
    CR_EXPECTS(!missingCount, "Missing Clang option value", Failure, CLANGRTC_INVALID_OPTION);

    for (auto *arg : parsed)
    {
        auto option = arg->getOption();
        auto name = option.getName();
        bool allowed = option.matches(clang::options::OPT_f_Group) || option.matches(clang::options::OPT_Diag_Group) ||
                       option.matches(clang::options::OPT_Preprocessor_Group) ||
                       option.matches(clang::options::OPT_IncludePath_Group) ||
                       option.matches(clang::options::OPT_g_Group) || name == "std=";

        for (auto prefix : {"fplugin", "fpass-plugin", "fmodule", "fprebuilt-module", "fcuda", "foffload", "fopenmp",
                            "fopenacc", "fsycl", "flto", "fembed", "fsyntax-only", "ftime-trace", "frewrite",
                            "fprofile", "fcoverage", "ftest-coverage", "fsave-optimization-record"})
        {
            if (name.starts_with(prefix))
            {
                allowed = false;
            }
        }

        if (option.matches(clang::options::OPT_Action_Group) || option.getKind() == llvm::opt::Option::InputClass ||
            option.getKind() == llvm::opt::Option::UnknownClass)
        {
            allowed = false;
        }

        CR_EXPECTS(allowed, std::format("Clang option is reserved or unsupported: {}", arg->getAsString(parsed)),
                   Failure, CLANGRTC_INVALID_OPTION);
    }
}

/**
 * @brief Parses the explicit option allowlist and normalizes its values.
 *
 * Options are processed in order. Path arguments become absolute, and
 * compute_NN targets are normalized to sm_NN. Unknown options and multiple
 * explicit output selectors are rejected, including repeated identical selectors.
 *
 * @param opts Complete option strings for this operation.
 * @return Validated settings for the requested operation.
 * @throws Failure for invalid options or unsupported output formats.
 */
Settings parse(std::vector<std::string> const &opts)
{
    Settings s;
    bool explicitOutput = false;
    bool deviceSource = false;
    bool gpuTarget = false;
    std::vector<std::string> clangOptions;

    for (size_t i = 0; i < opts.size(); ++i)
    {
        auto const &x = opts[i];
        // Extract an option value after its already-matched prefix.
        auto value = [&](std::string_view prefix) { return x.substr(prefix.size()); };

        if (x.starts_with("--clang-option=") || x == "--clang-option" || x.starts_with("--nvvm-option=") ||
            x == "--nvvm-option")
        {
            bool frontend = x.starts_with("--clang-option");
            std::string argument;
            auto equal = x.find('=');

            if (equal != std::string::npos)
            {
                argument = x.substr(equal + 1);
            }
            else
            {
                CR_EXPECTS(++i < opts.size(), "Missing pass-through option", Failure, CLANGRTC_INVALID_OPTION);
                argument = opts[i];
            }

            CR_EXPECTS(!argument.empty() && argument.find('\0') == std::string::npos,
                       "Empty or NUL-containing pass-through option", Failure, CLANGRTC_INVALID_OPTION);

            if (frontend)
            {
                clangOptions.push_back(argument);
                s.flags.push_back(argument);
            }
            else
            {
                auto key = argument.substr(0, argument.find('='));
                CR_EXPECTS(argument.starts_with("-") && key != "-arch" && key != "--gpu-architecture" &&
                               key != "-gen-lto" && key != "-nvvm-latest",
                           std::format("NVVM option is reserved or invalid: {}", argument), Failure,
                           CLANGRTC_INVALID_OPTION);
                s.nvvmOptions.push_back(argument);
            }
        }
        else if (x == "--require-kernel")
        {
            s.requireKernel = true;
        }
        else if (x == "--minimal")
        {
            s.minimal = true;
        }
        else if (x.starts_with("--stdlib="))
        {
            auto library = value("--stdlib=");
            CR_EXPECTS(library == "libcudacxx" || library == "host", "Expected --stdlib=host or --stdlib=libcudacxx",
                       Failure, CLANGRTC_INVALID_OPTION);
            s.libcudacxx = library == "libcudacxx";
        }
        else if (x == "--host-only")
        {
            s.device = false;
        }
        else if (x == "--source=c" || x == "--source=c++")
        {
            s.device = false;
            s.language = value("--source=");
        }
        else if (x.starts_with("--gpu-architecture="))
        {
            gpuTarget = true;
            s.target = value("--gpu-architecture=");
        }
        else if (x.starts_with("--cuda-gpu-arch="))
        {
            gpuTarget = true;
            s.target = value("--cuda-gpu-arch=");
        }
        else if (x.starts_with("--profile="))
        {
            s.profile = value("--profile=");
        }
        else if (x == "--ptx-text" || x == "--lto-ir" || x.starts_with("--emit="))
        {
            CR_EXPECTS(!explicitOutput, std::format("Only one output selector may be specified; got {}", x), Failure,
                       CLANGRTC_INVALID_OPTION);
            s.format = x.starts_with("--emit=") ? value("--emit=") : x.substr(2);
            explicitOutput = true;
        }
        else if (x.starts_with("--artifact="))
        {
            s.artifact = value("--artifact=");
        }
        else if (x.starts_with("--cuda-path="))
        {
            s.cuda = value("--cuda-path=");
        }
        else if (x.starts_with("--resource-dir="))
        {
            s.resource = value("--resource-dir=");
        }
        else if (x == "--source=preprocessed-cuda")
        {
            deviceSource = true;
            s.preprocessed = true;
        }
        else if (x == "--source=cuda")
        {
            deviceSource = true;
            s.preprocessed = false;
        }
        else if (x == "--time-trace")
        {
            s.trace = true;
        }
        else if (x.starts_with("--std="))
        {
            s.flags.push_back(std::format("-std={}", value("--std=")));
        }
        else if (x == "--device-int128")
        {
            s.flags.push_back("-fforce-enable-int128");
        }
        else if (x == "--fmad=true")
        {
            s.flags.push_back("-ffp-contract=fast");
        }
        else if (x == "--fmad=false")
        {
            s.flags.push_back("-ffp-contract=off");
        }
        else if (x == "-I" || x == "-isystem" || x == "-include" || x == "-D" || x == "-U")
        {
            CR_EXPECTS(++i < opts.size(), std::format("Missing value for {}", x), ::cr::Failure,
                       CLANGRTC_INVALID_OPTION);
            s.flags.push_back(x);
            s.flags.push_back((x == "-I" || x == "-isystem" || x == "-include") ? path(opts[i]) : opts[i]);
        }
        else if (x.starts_with("-I") && x.size() > 2)
        {
            s.flags.push_back(std::format("-I{}", path(x.substr(2))));
        }
        else if ((x.starts_with("-D") || x.starts_with("-U")) && x.size() > 2)
        {
            s.flags.push_back(x);
        }
        else if (x.starts_with("-std=") || x == "-w" || x == "-Werror" || x == "-g" || x == "-gline-tables-only" ||
                 x == "-fno-exceptions" || x == "-fno-rtti" || x == "-ffast-math")
        {
            s.flags.push_back(x);
        }
        else if (x.size() == 3 && x.starts_with("-O") && x[2] >= '0' && x[2] <= '3')
        {
            s.optimization = x[2] - '0';
        }
        else
        {
            CR_FAIL(std::format("Unsupported option: {}", x), ::cr::Failure, CLANGRTC_INVALID_OPTION);
        }
    }

    validate_clang_options(clangOptions);

    // The virtual filesystem starts at /. Normalize path-valued arguments just as
    // the named include options do, before saving frontend artifact identity.
    for (size_t i = 0; i < s.flags.size(); ++i)
    {
        if (s.flags[i] == "-I" || s.flags[i] == "-isystem" || s.flags[i] == "-include" || s.flags[i] == "-iquote" ||
            s.flags[i] == "-idirafter" || s.flags[i] == "-imacros")
        {
            CR_EXPECTS(i + 1 < s.flags.size(), "Missing Clang include path", Failure, CLANGRTC_INVALID_OPTION);
            ++i;
            s.flags[i] = path(s.flags[i]);
        }
        else if (s.flags[i].starts_with("-I") && s.flags[i].size() > 2)
        {
            s.flags[i] = "-I" + path(s.flags[i].substr(2));
        }
    }

    if (!s.device)
    {
        CR_EXPECTS(!s.minimal && !s.libcudacxx, "Device header modes require CUDA device compilation", Failure,
                   CLANGRTC_INVALID_OPTION);
        CR_EXPECTS(!s.requireKernel, "Kernel entry validation requires device mode", Failure, CLANGRTC_INVALID_OPTION);
        CR_EXPECTS(s.nvvmOptions.empty(), "NVVM options require device compilation", Failure, CLANGRTC_INVALID_OPTION);
        CR_EXPECTS(!deviceSource && !gpuTarget, "Host compilation cannot select CUDA source or a GPU target", Failure,
                   CLANGRTC_INVALID_OPTION);

        if (!explicitOutput)
        {
            s.format = "llvm-ir-bitcode";
        }

        CR_EXPECTS(s.format == "llvm-ir-bitcode" || s.format == "llvm-ir-text" || s.format == "object",
                   "Host compilation requires llvm-ir-bitcode, llvm-ir-text, or object output", Failure,
                   CLANGRTC_INVALID_OPTION);
        return s;
    }

    if (s.target.starts_with("compute_"))
    {
        s.target = std::format("sm_{}", s.target.substr(8));
    }

    CR_EXPECTS(s.target.starts_with("sm_") && s.target.size() > 3, "Expected GPU architecture sm_NN or compute_NN",
               ::cr::Failure, CLANGRTC_INVALID_OPTION);
    CR_EXPECTS(s.profile == "nvrtc" || s.profile == "hosted", "Unknown profile", ::cr::Failure,
               CLANGRTC_INVALID_OPTION);
    CR_EXPECTS(s.format == "nvvm-ir-bitcode" || s.format == "nvvm-ir-text" || s.format == "lto-ir" ||
                   s.format == "ptx-text",
               "Unsupported output format", ::cr::Failure, CLANGRTC_UNSUPPORTED);
    CR_EXPECTS(s.artifact == "pch" || s.artifact == "bmi" || s.artifact == "reduced-bmi" || s.artifact == "cuda" ||
                   s.artifact == "preprocessed-cuda",
               "Unknown artifact format", ::cr::Failure, CLANGRTC_INVALID_OPTION);
    return s;
}

/**
 * @brief Records dependency bytes while forwarding virtual filesystem reads.
 *
 * Successful file opens record readable contents and timestamps under the
 * requested path and any resolved real-path alias. The destination registry
 * must outlive this filesystem.
 */
struct RecordFS : llvm::vfs::ProxyFileSystem
{
    /**
     * @brief Borrowed registry populated with opened dependency files.
     */
    Files &files;

    /**
     * @brief Wraps a filesystem with dependency recording.
     *
     * @param fs Reference-counted underlying filesystem.
     * @param files Caller-owned destination for dependency snapshots.
     */
    RecordFS(llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs, Files &files)
        : ProxyFileSystem(std::move(fs)), files(files)
    {
    }

    /**
     * @brief Opens a file and snapshots its contents when status and buffer reads succeed.
     *
     * @param name Requested path, normalized for dependency recording.
     * @return Underlying open file, or its filesystem error if opening fails.
     */
    llvm::ErrorOr<std::unique_ptr<llvm::vfs::File>> openFileForRead(llvm::Twine const &name) override
    {
        auto file = getUnderlyingFS().openFileForRead(name);

        if (!file)
        {
            return file.getError();
        }

        auto status = (*file)->status();
        auto buffer = (*file)->getBuffer(name);

        if (status && buffer)
        {
            auto timestamp =
                std::chrono::duration_cast<std::chrono::seconds>(status->getLastModificationTime().time_since_epoch())
                    .count();
            auto contents = (*buffer)->getBuffer().str();
            files[path(name.str())] = {timestamp, contents};
            // Clang may reopen the same dependency through its resolved path when loading a PCH.
            llvm::SmallString<256> real;

            if (!getUnderlyingFS().getRealPath(name, real))
            {
                files[path(real.str().str())] = {timestamp, contents};
            }
        }

        return std::move(*file);
    }
};

} // namespace cr

/**
 * @brief Owns one operation's output, diagnostics, names, and timing views.
 *
 * Public views borrow this object's storage and can outlive the compiler.
 * Symbol and timing arrays are materialized once after the operation completes.
 */
struct clangrtcResultImpl
{
    /**
     * @brief Completion status of the prepare or compile operation.
     */
    clangrtcStatus status = CLANGRTC_SUCCESS;

    /**
     * @brief Format tag describing output bytes.
     */
    std::string format;

    /**
     * @brief Owned serialized artifact or LLVM IR output.
     */
    std::string output;

    /**
     * @brief Accumulated diagnostics and optional time-trace JSON.
     */
    std::string log;

    /**
     * @brief Owned requested expressions paired with their resolved symbols.
     */
    std::vector<std::pair<std::string, std::string>> names;

    /**
     * @brief Qualified CUDA kernel source names keyed by resolved symbol.
     */
    std::map<std::string, std::string> cudaNames;

    /**
     * @brief Private copy of the input artifact's slots, retained separately from this request.
     */
    std::vector<std::pair<std::string, std::string>> artifactNames;

    /**
     * @brief Stable public views borrowing expression and symbol strings from names.
     */
    std::vector<clangrtcSymbol> symbols;

    /**
     * @brief Owned stage names paired with elapsed nanoseconds.
     */
    std::vector<std::pair<std::string, uint64_t>> measurements;

    /**
     * @brief Stable public views borrowing stage names from measurements.
     */
    std::vector<clangrtcTiming> timings;

    /**
     * @brief Total operation duration in nanoseconds.
     */
    uint64_t total = 0;

    /** @brief Elapsed frontend work, including setup and failed attempts. */
    uint64_t frontendNs{};

    /**
     * @brief Whether public symbol and timing arrays have been materialized.
     */
    bool viewsReady = false;
};

/**
 * @brief Owns a loaded artifact and its self-contained virtual file collection.
 *
 * Metadata and dependency storage are populated before publication, then
 * remain unchanged while public views or compilation requests borrow them.
 */
struct clangrtcArtifactImpl
{
    /**
     * @brief Tagged artifact format: source, preprocessed source, PCH, or BMI.
     */
    std::string kind;

    /**
     * @brief GPU target used when preparing the artifact.
     */
    std::string target;

    /**
     * @brief Named-module identity, empty for non-BMI artifacts.
     */
    std::string module;

    /**
     * @brief Exact compiler fingerprint stored in the package.
     */
    std::string compiler = cr::fingerprint();

    /**
     * @brief Original ordered preparation options.
     */
    std::vector<std::string> options;

    /**
     * @brief Saved cc1 arguments for reconstructing a frontend invocation.
     */
    std::vector<std::string> args;

    /**
     * @brief Preparation expressions paired with resolved symbols in artifact slot order.
     */
    std::vector<std::pair<std::string, std::string>> names;

    /**
     * @brief Prepared CUDA kernel source names keyed by resolved symbol.
     */
    std::map<std::string, std::string> cudaNames;

    /**
     * @brief Stable public metadata views borrowing expression and symbol strings from names.
     */
    std::vector<clangrtcSymbol> symbols;

    /**
     * @brief Owned dependency snapshots, including virtual-path aliases.
     */
    cr::Files files;

    /**
     * @brief Primary source or serialized AST payload, interpreted according to kind.
     */
    std::string payload;

    /**
     * @brief Optional PCH restoring the header environment before a BMI import.
     */
    std::string bootstrap;

    /**
     * @brief LLVM bitcode containing prepared entries and their reachable implementation dependencies.
     */
    std::string ir;

    /**
     * @brief Virtual filename of the original preparation source.
     */
    std::string source;

    /**
     * @brief Content-derived virtual path for the primary payload.
     */
    std::string payloadPath;

    /**
     * @brief Content-derived virtual path for the bootstrap PCH.
     */
    std::string bootstrapPath;

    /**
     * @brief Total serialized package size in bytes.
     */
    size_t size = 0;

    /** @brief Validated package content identity, reused without hashing during compilation. */
    std::string contentHash;

    /**
     * @brief Public metadata views borrowing strings from options.
     */
    std::vector<clangrtcString> optionViews;

    /**
     * @brief Public metadata views borrowing dependency paths from files.
     */
    std::vector<clangrtcString> dependencyViews;

    /**
     * @brief In-memory filesystem containing dependencies and serialized AST payloads.
     */
    llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> memory;

    /**
     * @brief Reconstructs the virtual filesystem and borrowed metadata arrays.
     *
     * Call only after populating owned metadata and before publishing views.
     * Payload paths include content hashes so distinct artifacts do not collide
     * in Clang's module lookup. Rebuilding invalidates earlier metadata views.
     */
    void rebuild()
    {
        memory = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
        memory->setCurrentWorkingDirectory("/");
        dependencyViews.clear();

        for (auto const &[name, file] : files)
        {
            memory->addFile(name, file.first, llvm::MemoryBuffer::getMemBufferCopy(file.second, name));
            dependencyViews.push_back(cr::to_view(name));
        }

        // Content-derived paths keep distinct serialized ASTs from sharing a module-cache identity.
        payloadPath = std::format("/__clangrtc__/{}/module.pch", contentHash);
        bootstrapPath = std::format("/__clangrtc__/{}/bootstrap.pch", contentHash);

        if (!payload.empty())
        {
            memory->addFile(payloadPath, 0, llvm::MemoryBuffer::getMemBufferCopy(payload));
        }

        if (!bootstrap.empty())
        {
            memory->addFile(bootstrapPath, 0, llvm::MemoryBuffer::getMemBufferCopy(bootstrap));
        }

        symbols.clear();

        for (auto const &[expression, name] : names)
        {
            symbols.push_back({cr::to_view(expression), cr::to_view(name), cr::to_view(cudaNames[name])});
        }

        optionViews.clear();

        for (auto const &o : options)
        {
            optionViews.push_back(cr::to_view(o));
        }
    }
};

namespace cr
{
/**
 * @brief Bridges Clang diagnostics to the result log and public callback.
 *
 * The result and callback context are borrowed for this operation only.
 */
struct Diagnostics : clang::DiagnosticConsumer
{
    /**
     * @brief Optional synchronous callback supplied for this operation.
     */
    clangrtcDiagnosticCallback callback;

    /**
     * @brief Caller-owned context borrowed until this operation returns.
     */
    void *user;

    /**
     * @brief Borrowed operation result receiving diagnostics.
     */
    clangrtcResult result;

    /**
     * @brief Attaches diagnostic delivery to an operation.
     *
     * @param result Result accumulating diagnostic text.
     * @param callback Callback receiving this call's diagnostics, or NULL.
     * @param user Caller context forwarded unchanged to the callback.
     */
    Diagnostics(clangrtcResult result, clangrtcDiagnosticCallback callback, void *user)
        : callback(callback), user(user), result(result)
    {
    }

    /**
     * @brief Appends a diagnostic to the result and invokes the configured callback.
     *
     * Location prefixes are added to the log only when a filename is present.
     * Callback views borrow the arguments and remain valid only during the callback.
     *
     * @param text Diagnostic message without a location prefix.
     * @param stage Producing stage label.
     * @param level Public diagnostic severity.
     * @param file Optional source filename.
     * @param line One-based source line, or zero when unavailable.
     * @param col One-based source column, or zero when unavailable.
     */
    void message(std::string_view text, std::string_view stage, clangrtcDiagnosticLevel level, std::string_view file,
                 unsigned line, unsigned col)
    {
        if (!file.empty())
        {
            result->log.append(std::format("{}:{}:{}: ", file, line, col));
        }

        result->log.append(std::format("{}\n", text));

        if (callback)
        {
            clangrtcDiagnostic d{level, to_view(stage), to_view(text), to_view(file), line, col};
            callback(&d, user);
        }
    }

    /**
     * @brief Formats a Clang diagnostic and preserves its available source location.
     *
     * @param level Clang severity, mapped to a public diagnostic level.
     * @param info Diagnostic message, arguments, and source information.
     */
    void HandleDiagnostic(clang::DiagnosticsEngine::Level level, clang::Diagnostic const &info) override
    {
        clang::DiagnosticConsumer::HandleDiagnostic(level, info);
        llvm::SmallString<256> text;
        info.FormatDiagnostic(text);
        std::string_view file;
        unsigned line = 0, column = 0;

        if (info.hasSourceManager() && info.getLocation().isValid())
        {
            auto p = info.getSourceManager().getPresumedLoc(info.getLocation());

            if (p.isValid())
            {
                file = p.getFilename();
                line = p.getLine();
                column = p.getColumn();
            }
        }

        // Generated aliases use their own filenames. Errors in the actual source,
        // including instantiated template bodies, take precedence over bad slots.
        if (level >= clang::DiagnosticsEngine::Error)
        {
            if (file.starts_with(expression_source_prefix) || llvm::StringRef(text).starts_with("Expression slot "))
            {
                if (result->status != CLANGRTC_COMPILATION_FAILED)
                {
                    result->status = CLANGRTC_INVALID_EXPRESSION;
                }
            }
            else
            {
                result->status = CLANGRTC_COMPILATION_FAILED;
            }
        }

        message(std::string_view{text.data(), text.size()}, "frontend",
                level >= clang::DiagnosticsEngine::Error     ? CLANGRTC_DIAGNOSTIC_ERROR
                : level == clang::DiagnosticsEngine::Warning ? CLANGRTC_DIAGNOSTIC_WARNING
                                                             : CLANGRTC_DIAGNOSTIC_NOTE,
                file, line, column);
    }
};

/**
 * @brief Instantiates requested entry expressions and records their device symbols.
 *
 * Generated decltype aliases identify the declarations selected by the
 * request. This consumer runs before the code generator's translation-unit
 * callback, allowing selected definitions to be emitted and retained.
 */
struct Roots : clang::ASTConsumer
{
    /** @brief Optional record of retention attributes added by this request, for persistent frontend cleanup. */
    std::vector<clang::Decl *> *addedUsedAttrs{};

    /**
     * @brief Borrowed frontend providing AST, semantic analysis, and diagnostics.
     */
    clang::CompilerInstance &ci;

    /**
     * @brief Borrowed code generator owned by the enclosing multiplex consumer.
     */
    clang::ASTConsumer *cg;

    /**
     * @brief Borrowed slot array receiving resolved names in its existing order.
     */
    std::vector<std::pair<std::string, std::string>> &names;

    /**
     * @brief Source names collected directly from resolved CUDA kernel declarations.
     */
    std::map<std::string, std::string> &cudaNames;

    /**
     * @brief Alias prefix selecting the artifact or compilation slot namespace.
     */
    std::string_view prefix;

    /**
     * @brief Whether every selected entry must be a CUDA kernel.
     */
    bool requireKernel;

    /**
     * @brief Binds one slot namespace to the frontend and its code generator.
     *
     * @param ci Compiler instance kept alive throughout AST consumption.
     * @param cg Code generator receiving instantiated definitions.
     * @param names Owned expression/name pairs that outlive this consumer.
     * @param cudaNames Owned source-name metadata that outlives this consumer.
     * @param prefix Generated alias prefix identifying this slot namespace.
     * @param requireKernel Whether entries must be CUDA kernels.
     */
    Roots(clang::CompilerInstance &ci, clang::ASTConsumer *cg, std::vector<std::pair<std::string, std::string>> &names,
          std::map<std::string, std::string> &cudaNames, std::string_view prefix, bool requireKernel)
        : ci(ci), cg(cg), names(names), cudaNames(cudaNames), prefix(prefix), requireKernel(requireKernel)
    {
    }

    /**
     * @brief Resolves entry aliases, instantiates device definitions, and records mangled names.
     *
     * Invalid expressions or host declarations emit Clang diagnostics.
     * Functions and variables are marked used before being forwarded to codegen.
     *
     * @param context Completed AST containing the generated expression aliases.
     */
    void HandleTranslationUnit(clang::ASTContext &context) override
    {
        if (ci.getDiagnostics().hasErrorOccurred())
        {
            return;
        }

        std::unique_ptr<clang::MangleContext> mangle(context.createMangleContext());

        for (size_t index = 0; index < names.size(); ++index)
        {
            clang::ValueDecl *named{};
            clang::SourceLocation location;

            if (!context.getLangOpts().CPlusPlus)
            {
                auto declarations = context.getTranslationUnitDecl()->lookup(&context.Idents.get(names[index].first));
                named =
                    declarations.isSingleResult() ? llvm::dyn_cast<clang::ValueDecl>(declarations.front()) : nullptr;

                if (!named)
                {
                    ci.getDiagnostics().Report(ci.getDiagnostics().getCustomDiagID(
                        clang::DiagnosticsEngine::Error, "Expression slot %0 (%1) must name a C function or variable"))
                        << index << names[index].first;
                    continue;
                }

                location = named->getLocation();
            }
            else
            {
                // Resolve only this operation's aliases. Prepared aliases may also be
                // present in a loaded PCH, but their slot numbers belong to the artifact.
                // Their already compiled implementations are linked separately.
                auto aliasName = std::format("{}{}", prefix, index);
                auto declarations = context.getTranslationUnitDecl()->lookup(&context.Idents.get(aliasName));
                auto *alias = declarations.isSingleResult() ? llvm::dyn_cast<clang::TypeAliasDecl>(declarations.front())
                                                            : nullptr;

                if (!alias)
                {
                    ci.getDiagnostics().Report(ci.getDiagnostics().getCustomDiagID(
                        clang::DiagnosticsEngine::Error, "Missing prepared or compilation expression slot %0"))
                        << aliasName;
                    continue;
                }

                auto *type = llvm::dyn_cast<clang::DecltypeType>(alias->getUnderlyingType().getTypePtr());
                auto *expr = type ? type->getUnderlyingExpr()->IgnoreParenImpCasts() : nullptr;

                if (auto *op = llvm::dyn_cast_or_null<clang::UnaryOperator>(expr);
                    op && op->getOpcode() == clang::UO_AddrOf)
                {
                    expr = op->getSubExpr()->IgnoreParenImpCasts();
                }

                auto *ref = llvm::dyn_cast_or_null<clang::DeclRefExpr>(expr);

                if (!ref)
                {
                    ci.getDiagnostics().Report(
                        alias->getLocation(),
                        ci.getDiagnostics().getCustomDiagID(
                            clang::DiagnosticsEngine::Error,
                            "Expression slot %0 (%1) must identify a function or device variable"))
                        << index << names[index].first;
                    continue;
                }

                named = ref->getDecl();
                location = ref->getLocation();
            }

            if (requireKernel)
            {
                auto *fn = llvm::dyn_cast<clang::FunctionDecl>(named);

                if (!fn || !fn->hasAttr<clang::CUDAGlobalAttr>())
                {
                    ci.getDiagnostics().Report(location, ci.getDiagnostics().getCustomDiagID(
                                                             clang::DiagnosticsEngine::Error,
                                                             "Expression slot %0 (%1) must identify a CUDA kernel"))
                        << index << names[index].first;
                    continue;
                }
            }

            bool defined = false;

            if (auto *fn = llvm::dyn_cast<clang::FunctionDecl>(named))
            {
                if (ci.getLangOpts().CUDAIsDevice && !fn->hasAttr<clang::CUDADeviceAttr>() &&
                    !fn->hasAttr<clang::CUDAGlobalAttr>())
                {
                    ci.getDiagnostics().Report(
                        location, ci.getDiagnostics().getCustomDiagID(clang::DiagnosticsEngine::Error,
                                                                      "Expression slot %0 (%1) names a host function"))
                        << index << names[index].first;
                    continue;
                }

                // decltype resolves the specialization without requiring its body. Force that body
                // into the AST before codegen handles the requested declaration.
                if (fn->getTemplateInstantiationPattern())
                {
                    ci.getSema().InstantiateFunctionDefinition(location, fn, true, true);
                }

                if (auto *definition = fn->getDefinition())
                {
                    named = definition;
                    defined = true;
                }
            }
            else if (auto *var = llvm::dyn_cast<clang::VarDecl>(named))
            {
                if (ci.getLangOpts().CUDAIsDevice && !var->hasAttr<clang::CUDADeviceAttr>() &&
                    !var->hasAttr<clang::CUDAConstantAttr>())
                {
                    ci.getDiagnostics().Report(
                        location, ci.getDiagnostics().getCustomDiagID(clang::DiagnosticsEngine::Error,
                                                                      "Expression slot %0 (%1) names a host variable"))
                        << index << names[index].first;
                    continue;
                }

                if (var->getTemplateInstantiationPattern())
                {
                    ci.getSema().InstantiateVariableDefinition(location, var, true, true);
                }

                if (auto *definition = var->getDefinition())
                {
                    named = definition;
                    defined = true;
                }
            }

            if (!defined)
            {
                // Diagnose inside the frontend instead of throwing: Clang needs its
                // normal action teardown to release parser-owned state safely.
                ci.getDiagnostics().Report(
                    location, ci.getDiagnostics().getCustomDiagID(clang::DiagnosticsEngine::Error,
                                                                  "Expression slot %0 (%1) has no definition"))
                    << index << names[index].first;
                continue;
            }

            // The entry may have no callers in this translation unit, but the API promises its symbol.
            if (!named->hasAttr<clang::UsedAttr>())
            {
                named->addAttr(clang::UsedAttr::CreateImplicit(context));

                if (addedUsedAttrs)
                {
                    addedUsedAttrs->push_back(named);
                }
            }
            cg->HandleTopLevelDecl(clang::DeclGroupRef(named));
            std::string symbol;
            llvm::raw_string_ostream stream(symbol);

            if (context.getLangOpts().CPlusPlus && mangle->shouldMangleDeclName(named))
            {
                mangle->mangleName(clang::GlobalDecl(named), stream);
            }
            else
            {
                stream << named->getName();
            }

            names[index].second = symbol;

            if (auto *fn = llvm::dyn_cast<clang::FunctionDecl>(named); fn && fn->hasAttr<clang::CUDAGlobalAttr>())
            {
                // Read the declaration rather than demangling: source identity survives LTO output.
                auto &cudaName = cudaNames[symbol];
                cudaName.clear();
                llvm::raw_string_ostream sourceName(cudaName);
                fn->getNameForDiagnostic(sourceName, context.getPrintingPolicy(), true);
            }
        }
    }
};

/**
 * @brief Emits LLVM IR while retaining and reflecting requested device entries.
 */
struct Emit : clang::EmitLLVMOnlyAction
{
    /**
     * @brief Borrowed result receiving entry names during AST consumption.
     */
    clangrtcResult result;

    /**
     * @brief Generated alias prefix for this operation's expression namespace.
     */
    std::string_view prefix;

    /**
     * @brief Whether every selected entry must be a CUDA kernel.
     */
    bool requireKernel;

    /**
     * @brief Creates an LLVM-emission action for one operation's expression slots.
     *
     * @param result Result that must outlive this action and its consumers.
     * @param prefix Alias prefix distinguishing preparation and compilation slots.
     * @param requireKernel Whether entries must be CUDA kernels.
     */
    Emit(clangrtcResult result, std::string_view prefix, bool requireKernel)
        : result(result), prefix(prefix), requireKernel(requireKernel)
    {
    }

    /**
     * @brief Orders entry resolution before the ordinary LLVM code generator.
     *
     * @param ci Compiler instance providing frontend state.
     * @param name Input filename forwarded to the LLVM-emission action.
     * @return Multiplex consumer owning the entry resolver and code generator.
     */
    std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(clang::CompilerInstance &ci, llvm::StringRef name) override
    {
        auto cg = clang::EmitLLVMOnlyAction::CreateASTConsumer(ci, name);
        auto *ptr = cg.get();
        std::vector<std::unique_ptr<clang::ASTConsumer>> consumers;
        // Resolve entries first so the code generator sees their instantiated definitions
        // before its own end-of-translation-unit callback.
        consumers.push_back(std::make_unique<Roots>(ci, ptr, result->names, result->cudaNames, prefix, requireKernel));
        consumers.push_back(std::move(cg));
        return std::make_unique<clang::MultiplexConsumer>(std::move(consumers));
    }
};

/**
 * @brief Builds driver arguments for device compilation or explicit artifact preparation.
 *
 * Selects toolkit headers, a GPU target, and the requested header profile.
 * The NVRTC profile supplies its bootstrap.
 * Named-module sources include their bootstrap in the global module fragment instead.
 *
 * @param s Parsed settings and ordered caller flags.
 * @param name Absolute virtual source filename.
 * @param mode Preparation format, or compile for normal LLVM emission.
 * @param inputs Request filesystem receiving device header feature gates.
 * @return Owned driver arguments, including the executable-name placeholder.
 * @throws Failure if the CUDA toolkit or its version definition cannot be found.
 */
std::vector<std::string> driverArgs(Settings const &s, std::string const &name, std::string const &mode,
                                    llvm::vfs::InMemoryFileSystem &inputs)
{
    if (!s.device)
    {
        auto resourceHeader = std::format("{}/include/stddef.h", s.resource);
        CR_EXPECTS(std::filesystem::exists(resourceHeader), "Missing Clang resource headers", Failure,
                   CLANGRTC_TOOLCHAIN_ERROR);
        std::vector<std::string> args{"clang++",
                                      "-x",
                                      s.language,
                                      "-target",
                                      llvm::sys::getDefaultTargetTriple(),
                                      "-resource-dir",
                                      s.resource,
                                      "-fPIC",
                                      s.language == "c" ? "-std=c17" : "-std=c++20",
                                      std::format("-O{}", s.optimization)};
        args.insert(args.end(), s.flags.begin(), s.flags.end());

        if (mode == "pch")
        {
            args.insert(args.end(), {"-S", "-Xclang", "-emit-pch"});
        }
        else if (mode == "bmi" || mode == "reduced-bmi")
        {
            args.insert(args.end(),
                        {"-S", "-Xclang", mode == "bmi" ? "-emit-module-interface" : "-emit-reduced-module-interface"});
        }
        else
        {
            args.insert(args.end(), {"-S", "-emit-llvm"});
        }

        args.insert(args.end(), {name, "-o", "/__clangrtc__/output"});
        return args;
    }

    auto cuda = s.cuda;

    if (cuda.empty())
    {
        if (auto *p = std::getenv("CUDA_PATH"))
        {
            cuda = p;
        }
        else if (auto *p = std::getenv("CONDA_PREFIX"))
        {
            cuda = p;
        }
    }

    CR_EXPECTS(!cuda.empty(), "Specify --cuda-path=TOOLKIT or CUDA_PATH", Failure, CLANGRTC_TOOLCHAIN_ERROR);
    auto resourceHeader = std::format("{}/include/stddef.h", s.resource);
    std::ifstream resource(resourceHeader);
    CR_EXPECTS(static_cast<bool>(resource), std::format("Cannot read Clang resource header {}", resourceHeader),
               Failure, CLANGRTC_TOOLCHAIN_ERROR);
    auto include = std::format("{}/include", cuda);
    std::error_code error;

    if (!std::filesystem::exists(std::format("{}/cuda_runtime_api.h", include), error))
    {
        include = std::format("{}/targets/{}-linux/include", cuda,
                              llvm::sys::getDefaultTargetTriple().starts_with("aarch64") ? "aarch64" : "x86_64");
    }

    std::string version = read(std::format("{}/cuda_runtime_api.h", include));
    auto at = version.find("#define CUDART_VERSION");
    CR_EXPECTS(at != std::string::npos,
               std::format("Toolkit header {}/cuda_runtime_api.h does not define CUDART_VERSION", include), Failure,
               CLANGRTC_TOOLCHAIN_ERROR);
    at += std::string_view("#define CUDART_VERSION").size();
    auto number = llvm::StringRef(version).substr(at).ltrim(" \t").take_while(llvm::isDigit);
    int v = 0;
    CR_EXPECTS(!number.empty() && !number.getAsInteger(10, v),
               std::format("Invalid CUDART_VERSION in {}/cuda_runtime_api.h", include), Failure,
               CLANGRTC_TOOLCHAIN_ERROR);

    if (s.minimal || s.libcudacxx)
    {
        // Overlay the genuine compiler/SDK headers at their original paths. This preserves
        // system-header handling and intercepts SDK quoted includes as well as angle includes.
        auto overlay = [&](std::string const &name, std::string_view contents) {
            if (!inputs.exists(name))
            {
                CR_EXPECTS(inputs.addFile(name, 0, llvm::MemoryBuffer::getMemBufferCopy(contents, name)),
                           std::format("Cannot configure CUDA header {}", name));
            }
        };

        for (auto const &[name, contents] : cuda_mode_headers())
        {
            overlay(std::format("{}/include/{}", s.resource, name), contents);
        }

        if (s.minimal)
        {
            overlay(std::format("{}/cuda_runtime_api.h", include),
                    std::format("#ifndef __CUDACC_RTC_MINIMAL__\n{}\n#endif\n", version));
        }
    }

    std::vector<std::string> args{"clang++",
                                  "--cuda-device-only",
                                  std::format("--cuda-gpu-arch={}", s.target),
                                  std::format("--cuda-path={}", cuda),
                                  "-resource-dir",
                                  s.resource,
                                  "-nocudalib",
                                  "-std=c++20",
                                  std::format("-O{}", s.optimization),
                                  "-Wno-unknown-cuda-version",
                                  std::format("-I{}", include)};

    if (s.minimal || s.libcudacxx)
    {
        args.push_back(std::format("-DCUDA_VERSION={}", v));
    }

    if (s.minimal)
    {
        args.push_back("-D__CUDACC_RTC_MINIMAL__=1");
        args.push_back(std::format("-DCUDART_VERSION={}", v));
    }

    if (s.libcudacxx)
    {
        args.insert(args.end(),
                    {"-nostdinc++", "-fno-exceptions", "-D__CLANGRTC_LIBCUDACXX__=1", "-D_CCCL_ENABLE_FREESTANDING=1",
                     "-I/__clangrtc__/include/clangrtc/libcudacxx", std::format("-idirafter{}/cccl", include)});
        // cuda_runtime.h's convenience templates include <utility> before CUDA's address
        // space intrinsics are ready. Keep its C API and load its C++ adapters afterwards.
        auto runtime = read(std::format("{}/cuda_runtime.h", include));
        llvm::StringRef condition = "defined(__cplusplus) && !defined(__CUDACC_RTC__)";
        size_t at{};

        while ((at = runtime.find(condition.str(), at)) != std::string::npos)
        {
            runtime.insert(at + condition.size(), " && !defined(__CLANGRTC_LIBCUDACXX__)");
            at += condition.size() + std::string_view(" && !defined(__CLANGRTC_LIBCUDACXX__)").size();
        }

        auto name = std::format("{}/cuda_runtime.h", include);

        if (!inputs.exists(name))
        {
            CR_EXPECTS(inputs.addFile(name, 0, llvm::MemoryBuffer::getMemBufferCopy(runtime, name)),
                       "Cannot configure CUDA runtime header");
        }
    }

    args.push_back("-I/__clangrtc__/include");

    // CUDA's implicit wrapper precedes the source, but a module's global fragment must come first.
    // Module producers include clangrtc/device.hpp explicitly after `module;` instead.
    if (mode == "bmi" || mode == "reduced-bmi")
    {
        args.push_back("-nocudainc");
    }

    args.insert(args.end(), s.flags.begin(), s.flags.end());

    if (mode == "pch")
    {
        args.insert(args.end(), {"-x", "cuda", "-S", "-Xclang", "-emit-pch"});
    }
    else if (mode == "bmi" || mode == "reduced-bmi")
    {
        args.insert(args.end(), {"-x", "cuda", "-S", "-Xclang",
                                 mode == "bmi" ? "-emit-module-interface" : "-emit-reduced-module-interface"});
    }
    else if (mode == "preprocessed-cuda")
    {
        args.insert(args.end(), {"-x", "cuda", "-E"});
    }
    else
    {
        args.insert(args.end(), {"-x", "cuda", "-S", "-emit-llvm"});
    }

    args.push_back(name);
    args.insert(args.end(), {"-o", "/__clangrtc__/output"});
    return args;
}

/**
 * @brief Constructs a frontend invocation from driver arguments or saved cc1 arguments.
 *
 * With saved output requested, the in-process driver chooses a cc1 job
 * and its arguments are copied for later reuse. Saved cc1 arguments bypass
 * driver reconstruction. The disable-free flag is omitted so frontend state
 * is released when its compiler instance is destroyed.
 *
 * @param args Driver arguments when saved is non-NULL; otherwise saved cc1 arguments.
 * @param fs Filesystem visible to driver input discovery.
 * @param diagnostics Operation-local result and callback configuration.
 * @param saved Optional destination for the selected cc1 argument vector.
 * @return Owned frontend invocation ready for operation-specific adjustments.
 * @throws Failure if driver construction or frontend option parsing fails.
 */
std::shared_ptr<clang::CompilerInvocation> invocation(std::vector<std::string> const &args,
                                                      llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs,
                                                      Diagnostics &diagnostics, std::vector<std::string> *saved)
{
    Diagnostics diag(diagnostics.result, diagnostics.callback, diagnostics.user);
    clang::CompilerInstance parser;
    parser.createDiagnostics(&diag, false);
    std::vector<std::string> cc1;

    if (saved)
    {
        auto directory = fs->getCurrentWorkingDirectory();
        CR_EXPECTS(directory, "Cannot determine compiler filesystem directory", ::cr::Failure,
                   CLANGRTC_TOOLCHAIN_ERROR);
        // The driver discovers installed SDKs and host toolchains. The frontend still uses fs below,
        // so a prepared artifact's source/header snapshot remains authoritative during compilation.
        auto driverFs = llvm::makeIntrusiveRefCnt<llvm::vfs::OverlayFileSystem>(llvm::vfs::getRealFileSystem());
        driverFs->pushOverlay(fs);
        // pushOverlay synchronizes its child's cwd; restore the frontend snapshot before continuing.
        auto directoryError = fs->setCurrentWorkingDirectory(*directory);
        CR_EXPECTS(!directoryError,
                   std::format("Cannot restore compiler filesystem directory: {}", directoryError.message()),
                   ::cr::Failure, CLANGRTC_TOOLCHAIN_ERROR);
        clang::driver::Driver driver("clang++", llvm::sys::getDefaultTargetTriple(), parser.getDiagnostics(),
                                     "clangrtc", driverFs);
        std::vector<char const *> p;

        for (auto const &s : args)
        {
            p.push_back(s.c_str());
        }

        std::unique_ptr<clang::driver::Compilation> job(driver.BuildCompilation(p));
        CR_EXPECTS(job && !parser.getDiagnostics().hasErrorOccurred(), "Invalid compiler options", ::cr::Failure,
                   CLANGRTC_INVALID_OPTION);

        for (auto const &command : job->getJobs())
        {
            if (command.getCreator().getName() == llvm::StringRef("clang"))
            {
                auto const &a = command.getArguments();

                if (!a.empty() && llvm::StringRef(a[0]) == "-cc1")
                {
                    for (size_t i = 1; i < a.size(); ++i)
                    {
                        // The driver assumes a short-lived process; this library must release
                        // frontend allocations after every request.
                        if (llvm::StringRef(a[i]) != "-disable-free")
                        {
                            cc1.emplace_back(a[i]);
                        }
                    }

                    break;
                }
            }
        }

        CR_EXPECTS(!cc1.empty(), "Clang driver did not produce a frontend command", ::cr::Failure,
                   CLANGRTC_INVALID_OPTION);
        *saved = cc1;
    }
    else
    {
        cc1 = args;
    }

    auto result = std::make_shared<clang::CompilerInvocation>();
    std::vector<char const *> p;

    for (auto const &s : cc1)
    {
        p.push_back(s.c_str());
    }

    CR_EXPECTS(clang::CompilerInvocation::CreateFromArgs(*result, p, parser.getDiagnostics()),
               "Invalid frontend invocation", ::cr::Failure, CLANGRTC_INVALID_OPTION);
    return result;
}

/**
 * @brief Packages artifact metadata and binary payloads with an integrity digest.
 *
 * The package contains a versioned header, JSON manifest length, XXH3-128
 * digest, manifest, and concatenated payload/dependency bytes. The digest
 * checks integrity; it does not authenticate the producer.
 *
 * @param a Artifact with populated owned metadata and payloads.
 * @return Owned serialized package suitable for loading from memory.
 */
std::string serialize(clangrtcArtifactImpl const &a)
{
    llvm::json::Object meta;
    meta["compiler"] = a.compiler;
    meta["kind"] = a.kind;
    meta["target"] = a.target;
    meta["module"] = a.module;
    meta["source"] = a.source;
    meta["payload"] = a.payload.size();
    meta["bootstrap"] = a.bootstrap.size();
    meta["ir"] = a.ir.size();
    llvm::json::Array options, args, files, symbols;

    for (auto const &[expression, name] : a.names)
    {
        llvm::json::Object symbol{{"expression", expression}, {"name", name}};

        if (auto source = a.cudaNames.find(name); source != a.cudaNames.end())
        {
            symbol["cuda_name"] = source->second;
        }

        symbols.push_back(std::move(symbol));
    }

    meta["symbols"] = std::move(symbols);

    for (auto const &x : a.options)
    {
        options.push_back(x);
    }

    for (auto const &x : a.args)
    {
        args.push_back(x);
    }

    // Lengths in the manifest delimit binary sections; neither ASTs nor source files
    // need separators or NUL terminators in the package body.
    std::string body = a.payload;
    body.append(a.bootstrap);
    body.append(a.ir);

    for (auto const &[name, file] : a.files)
    {
        llvm::json::Object f;
        f["path"] = name;
        f["mtime"] = file.first;
        f["size"] = file.second.size();
        files.push_back(std::move(f));
        body.append(file.second);
    }

    meta["options"] = std::move(options);
    meta["args"] = std::move(args);
    meta["files"] = std::move(files);
    std::string json;
    llvm::raw_string_ostream os(json);
    os << llvm::json::Value(std::move(meta));
    auto bytes = json;
    bytes.append(body);
    auto package = std::format("{}\n{}\n{}\n", artifact_package_magic, json.size(), hash(bytes));
    package.append(bytes);
    return package;
}

/**
 * @brief Validates a package and reconstructs its memory filesystem.
 *
 * Checks framing, checksum, compiler identity, lengths, dependency records,
 * and trailing bytes before publishing the artifact. Serialized Clang state
 * must still come from a trusted producer.
 *
 * @param bytes Package bytes, borrowed during this call.
 * @return Owned artifact with stable metadata views and an in-memory file collection.
 * @throws Failure for malformed packages or an incompatible compiler fingerprint.
 */
std::unique_ptr<clangrtcArtifactImpl> deserialize(std::span<char const> bytes)
{
    auto envelope = read_artifact_envelope(bytes);
    auto size = envelope.manifest_size;
    auto begin = bytes.size() - envelope.contents.size();
    // Verify both the manifest and body before trusting any recorded section lengths.
    CR_EXPECTS(hash(envelope.contents) == envelope.checksum, "Artifact checksum mismatch", Failure,
               CLANGRTC_INVALID_ARTIFACT);
    std::string_view manifest{bytes.data() + begin, size};
    auto parsed = llvm::json::parse(llvm::StringRef(manifest));

    if (!parsed)
    {
        llvm::consumeError(parsed.takeError());
        CR_FAIL("Invalid artifact manifest", Failure, CLANGRTC_INVALID_ARTIFACT);
    }

    auto *meta = parsed->getAsObject();
    CR_EXPECTS(meta, "Artifact manifest must be an object", Failure, CLANGRTC_INVALID_ARTIFACT);
    // Require a string-valued manifest field and return an owned copy.
    auto string = [&](char const *key) {
        auto v = meta->getString(key);
        CR_EXPECTS(v.has_value(), std::format("Missing {}", key), Failure, CLANGRTC_INVALID_ARTIFACT);
        return v->str();
    };

    auto a = std::make_unique<clangrtcArtifactImpl>();
    a->contentHash = envelope.checksum;
    a->compiler = string("compiler");
    CR_EXPECTS(a->compiler == fingerprint(), "Artifact compiler fingerprint differs", ::cr::Failure,
               CLANGRTC_INCOMPATIBLE_ARTIFACT);
    a->kind = string("kind");
    CR_EXPECTS(a->kind == "pch" || a->kind == "bmi" || a->kind == "reduced-bmi" || a->kind == "cuda" ||
                   a->kind == "preprocessed-cuda",
               "Invalid artifact kind", Failure, CLANGRTC_INVALID_ARTIFACT);
    a->target = string("target");
    a->module = string("module");
    a->source = string("source");
    begin += size;
    // Consume exactly one length-checked payload section from the package body.
    auto take = [&](int64_t n) {
        CR_EXPECTS(n >= 0 && static_cast<uint64_t>(n) <= bytes.size() - begin, "Truncated artifact data", Failure,
                   CLANGRTC_INVALID_ARTIFACT);
        auto section = bytes.subspan(begin, n);
        auto value = std::string(section.data(), section.size());
        begin += n;
        return value;
    };

    auto payload = meta->getInteger("payload"), bootstrap = meta->getInteger("bootstrap");
    CR_EXPECTS(payload && bootstrap, "Missing artifact lengths", Failure, CLANGRTC_INVALID_ARTIFACT);
    a->payload = take(*payload);
    a->bootstrap = take(*bootstrap);

    // Packages written before preparation slots did not contain an IR section.
    if (auto *section = meta->get("ir"))
    {
        auto size = section->getAsInteger();
        CR_EXPECTS(size.has_value(), "Invalid prepared IR length", Failure, CLANGRTC_INVALID_ARTIFACT);
        a->ir = take(*size);
    }

    // Decode a required array of strings without changing their order.
    auto list = [&](char const *key, std::vector<std::string> &result) {
        auto *xs = meta->getArray(key);
        CR_EXPECTS(xs, "Missing list", Failure, CLANGRTC_INVALID_ARTIFACT);

        for (auto const &v : *xs)
        {
            auto s = v.getAsString();
            CR_EXPECTS(s.has_value(), "Invalid string list", Failure, CLANGRTC_INVALID_ARTIFACT);
            result.push_back(s->str());
        }
    };

    list("options", a->options);
    list("args", a->args);

    // Older packages have no preparation slots. Absence means an empty namespace.
    if (auto *symbols = meta->get("symbols"))
    {
        auto *entries = symbols->getAsArray();
        CR_EXPECTS(entries && entries->size() < entry_count_limit, "Invalid artifact expression list", Failure,
                   CLANGRTC_INVALID_ARTIFACT);

        for (auto const &entry : *entries)
        {
            auto *slot = entry.getAsObject();
            CR_EXPECTS(slot, "Invalid artifact expression slot", Failure, CLANGRTC_INVALID_ARTIFACT);
            auto expression = slot->getString("expression"), name = slot->getString("name");
            CR_EXPECTS(expression && name && !name->empty(), "Invalid artifact expression metadata", Failure,
                       CLANGRTC_INVALID_ARTIFACT);
            a->names.emplace_back(expression->str(), name->str());

            if (auto *source = slot->get("cuda_name"))
            {
                auto value = source->getAsString();
                CR_EXPECTS(value.has_value(), "Invalid CUDA source name", Failure, CLANGRTC_INVALID_ARTIFACT);
                a->cudaNames[name->str()] = value->str();
            }
        }
    }

    CR_EXPECTS(a->names.empty() || !a->ir.empty(), "Missing prepared entry definitions", Failure,
               CLANGRTC_INVALID_ARTIFACT);
    auto *files = meta->getArray("files");
    CR_EXPECTS(files && files->size() < entry_count_limit, "Invalid dependency list", Failure,
               CLANGRTC_INVALID_ARTIFACT);

    for (auto const &entry : *files)
    {
        auto *f = entry.getAsObject();
        CR_EXPECTS(f, "Invalid dependency", Failure, CLANGRTC_INVALID_ARTIFACT);
        auto name = f->getString("path");
        auto time = f->getInteger("mtime"), n = f->getInteger("size");
        CR_EXPECTS(name && time && n, "Invalid dependency metadata", Failure, CLANGRTC_INVALID_ARTIFACT);
        CR_EXPECTS(!a->files.contains(name->str()), "Duplicate dependency", Failure, CLANGRTC_INVALID_ARTIFACT);
        a->files[name->str()] = {*time, take(*n)};
    }

    CR_EXPECTS(begin == bytes.size(), "Trailing artifact bytes", Failure, CLANGRTC_INVALID_ARTIFACT);
    a->size = bytes.size();
    a->rebuild();
    return a;
}

/**
 * @brief Builds the option identity used to compare prepared compiler state.
 *
 * Includes the target, header profile, optimization level, device header
 * configuration, and ordered frontend flags. Output-format, artifact-format,
 * and tracing selections do not contribute to this identity.
 *
 * @param s Settings whose prepared-state identity is requested.
 * @return Comparison string for the represented semantic settings.
 */
std::string semantic(Settings const &s)
{
    auto key = std::format("{}\n{}\n{}\n{}\n{}\n{}\n{}", s.device, s.language, s.target, s.profile, s.optimization,
                           s.minimal, s.libcudacxx);

    for (auto const &f : s.flags)
    {
        key.append(std::format("\n{}", f));
    }

    return key;
}

/**
 * @brief Copies one file into an in-memory filesystem.
 *
 * @param fs Destination filesystem.
 * @param name Virtual filename to add.
 * @param data File bytes copied into the filesystem.
 * @param time Modification time in seconds since the epoch; zero for generated inputs.
 * @throws Failure when the filesystem rejects the file insertion.
 */
void add(llvm::vfs::InMemoryFileSystem &fs, std::string const &name, std::string const &data, int64_t time)
{
    CR_EXPECTS(fs.addFile(name, time, llvm::MemoryBuffer::getMemBufferCopy(data, name)),
               std::format("Duplicate virtual file: {}", name));
}

/**
 * @brief Owns the mutable source, options, and filesystem state for one operation.
 *
 * Caller strings are copied before frontend work starts. The filesystem
 * overlays request files on either the host filesystem or an artifact's
 * packaged dependencies, while recording files opened by the frontend.
 * The artifact and diagnostic sink remain borrowed for the duration of the run.
 */
struct Run
{
    /**
     * @brief Borrowed operation-local diagnostic sink used while creating invocations.
     */
    Diagnostics &diagnostics;

    /**
     * @brief Borrowed result receiving names, diagnostics, and stage timings.
     */
    clangrtcResult result;

    /**
     * @brief Dependency snapshots recorded while reading the operation filesystem.
     */
    Files files;

    /**
     * @brief Owned request overlay containing source, virtual headers, and generated files.
     */
    llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> inputs =
        llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();

    /**
     * @brief Recording filesystem exposing the request overlay and its base.
     */
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs;

    /**
     * @brief Explicit call options, or the artifact's preparation options when omitted.
     */
    std::vector<std::string> options;

    /**
     * @brief Validated settings derived from effective options.
     */
    Settings settings;

    /**
     * @brief Absolute virtual filename assigned to the compilation source.
     */
    std::string sourceName;

    /**
     * @brief Owned source, including generated entry-expression aliases.
     */
    std::string source;

    /**
     * @brief Driver-selected cc1 arguments recorded when building a new invocation.
     */
    std::vector<std::string> args;

    /**
     * @brief Borrowed artifact supplying prepared state, or NULL for direct input.
     */
    clangrtcArtifact artifact;

    /**
     * @brief Validates a request and assembles its owned compilation inputs.
     *
     * Rejects incompatible artifact settings and replacement of prepared
     * dependencies. Entry expressions become decltype aliases for semantic
     * resolution during compilation.
     *
     * @param result Result receiving expression/name entries.
     * @param diagnostics Diagnostic sink that outlives all frontends in this run.
     * @param request Borrowed source, virtual headers, optional artifact, and expressions.
     * @param opts Complete option views, borrowed during construction.
     * @param count Number of options.
     * @param preparing Whether this run prepares an artifact rather than compiling a consumer.
     * @throws Failure if inputs, options, or prepared-state compatibility checks fail.
     */
    Run(clangrtcResult result, Diagnostics &diagnostics, clangrtcRequest const &request, clangrtcString const *opts,
        size_t count, bool preparing)
        : diagnostics(diagnostics), result(result), artifact(request.artifact)
    {
        CR_EXPECTS(!preparing || !artifact, "Preparation cannot consume an artifact");
        options = to_strings(opts, count);

        // With no explicit options, inherit the complete preparation environment.
        // Explicit options must describe the same semantic environment, checked below.
        if (artifact && options.empty())
        {
            options = artifact->options;
        }

        settings = parse(options);
        CR_EXPECTS(settings.device || !preparing || settings.artifact == "pch" || settings.artifact == "bmi" ||
                       settings.artifact == "reduced-bmi",
                   "Host preparation supports PCH, BMI, and reduced BMI", Failure, CLANGRTC_UNSUPPORTED);
        CR_EXPECTS(settings.device || !preparing || settings.language == "c++" || settings.artifact == "pch",
                   "BMI preparation requires C++ source", Failure, CLANGRTC_UNSUPPORTED);

        if (artifact)
        {
            CR_EXPECTS(settings.device || artifact->kind == "pch" || artifact->kind == "bmi" ||
                           artifact->kind == "reduced-bmi",
                       "Host compilation requires a PCH or BMI artifact", Failure, CLANGRTC_UNSUPPORTED);
            CR_EXPECTS(semantic(settings) == semantic(parse(artifact->options)),
                       "Semantic options differ from artifact", ::cr::Failure, CLANGRTC_INCOMPATIBLE_ARTIFACT);
        }

        inputs->setCurrentWorkingDirectory("/");

        if (!artifact)
        {
            for (auto const &[name, data] : builtin_headers())
            {
                add(*inputs, name, data, 0);
            }
        }

        source = to_string(request.source.data);
        sourceName = request.source.path.size ? path(to_string(request.source.path))
                     : preparing              ? "/__clangrtc__/prefix.cuh"
                                              : "/__clangrtc__/request.cu";

        if (artifact && (artifact->kind == "cuda" || artifact->kind == "preprocessed-cuda"))
        {
            if (source.empty())
            {
                source = artifact->payload;
                settings.preprocessed = artifact->kind == "preprocessed-cuda";
            }
            else
            {
                // A new consumer may contain directives and macros of its own. Use the
                // original prefix with its packaged headers rather than mixing raw
                // consumer tokens into an already expanded translation unit.
                auto prefix = artifact->files.find(artifact->source);
                CR_EXPECTS(prefix != artifact->files.end(), "Missing artifact source dependency", Failure,
                           CLANGRTC_INVALID_ARTIFACT);
                source = std::format("{}\n#line 1 \"{}\"\n{}", prefix->second.second,
                                     clang::Lexer::Stringify(sourceName), source);
            }
        }

        CR_EXPECTS(!source.empty() || preparing, "No compilation source");
        CR_EXPECTS(
            !artifact || !artifact->files.contains(sourceName) || artifact->files.at(sourceName).second == source,
            "Runtime source cannot replace an artifact dependency", ::cr::Failure, CLANGRTC_INCOMPATIBLE_ARTIFACT);
        CR_EXPECTS(request.headers || !request.headerCount, "NULL headers");
        CR_EXPECTS(request.headerCount < entry_count_limit, "Too many headers");

        for (size_t i = 0; i < request.headerCount; ++i)
        {
            auto name = path(to_string(request.headers[i].path));
            auto data = to_string(request.headers[i].data);

            if (artifact && artifact->files.contains(name))
            {
                CR_EXPECTS(artifact->files.at(name).second == data,
                           std::format("Cannot replace prepared dependency {}", name), ::cr::Failure,
                           CLANGRTC_INCOMPATIBLE_ARTIFACT);
            }
            else
            {
                add(*inputs, name, data, 0);
            }
        }

        // Let Clang resolve overloads and template arguments through ordinary C++ lookup.
        // Roots later maps each numbered alias back to the caller's expression slot.
        auto expressions = to_strings(request.expressions, request.expressionCount);
        auto prefix = preparing ? artifact_expression_prefix : compilation_expression_prefix;

        if (artifact)
        {
            result->artifactNames = artifact->names;
        }

        auto sourceLine = std::count(source.begin(), source.end(), '\n') + 1;
        auto quotedSourceName = clang::Lexer::Stringify(sourceName);

        for (size_t i = 0; i < expressions.size(); ++i)
        {
            result->names.emplace_back(expressions[i], "");
            auto expressionPath =
                std::format("{}{}-slot-{}.cu", expression_source_prefix, preparing ? "artifact" : "compilation", i);

            if (settings.device || settings.language != "c")
            {
                source.append(std::format("\n#line 1 \"{}\"\nusing {}{} = decltype({});\n#line {} \"{}\"\n",
                                          expressionPath, prefix, i, expressions[i], sourceLine, quotedSourceName));
            }
        }

        add(*inputs, sourceName, source, 0);
        // Prepared compilations read packaged dependencies, independent of the original disk files.
        // Only this request's source and additional headers sit above that immutable snapshot.
        auto base = artifact ? llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem>(artifact->memory)
                             : llvm::vfs::getRealFileSystem();
        auto overlay = llvm::makeIntrusiveRefCnt<llvm::vfs::OverlayFileSystem>(base);
        overlay->pushOverlay(inputs);
        // Prepared dependencies are already owned by the artifact. Recording their reads would
        // copy PCH/BMI buffers and header contents again on every consumer compilation.
        fs = artifact ? llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem>(overlay)
                      : llvm::makeIntrusiveRefCnt<RecordFS>(overlay, files);
    }

    /**
     * @brief Creates or restores an invocation and records its construction time.
     *
     * PCH/BMI consumers reuse packaged cc1 settings, replace the input, clear
     * module-production state, and attach the prepared PCH or named module.
     * Other inputs use the driver to construct a fresh invocation.
     *
     * @param mode Preparation format or compile action to select.
     * @return Invocation configured for the operation's source and prepared state.
     */
    std::shared_ptr<clang::CompilerInvocation> get(std::string const &mode)
    {
        auto start = Clock::now();
        CR_DEFER([&] { result->frontendNs += elapsed(start); });
        std::shared_ptr<clang::CompilerInvocation> inv;

        if (artifact && (artifact->kind == "pch" || artifact->kind == "bmi" || artifact->kind == "reduced-bmi"))
        {
            inv = invocation(artifact->args, fs, diagnostics, nullptr);
            // Reuse the prepared language/target settings, but switch from producing
            // a PCH or module to compiling this consumer translation unit.
            inv->getFrontendOpts().Inputs.clear();
            inv->getFrontendOpts().Inputs.emplace_back(
                sourceName, clang::InputKind(settings.device ? clang::Language::CUDA
                                                             : (settings.language == "c" ? clang::Language::C
                                                                                         : clang::Language::CXX)));
            inv->getFrontendOpts().ProgramAction = clang::frontend::EmitLLVMOnly;
            inv->getFrontendOpts().OutputFile.clear();
            inv->getLangOpts().setCompilingModule(clang::LangOptions::CMK_None);
            inv->getLangOpts().ModuleName.clear();
            inv->getLangOpts().CurrentModule.clear();
            auto &pp = inv->getPreprocessorOpts();
            // The serialized prefix already contains forced includes. Replaying them
            // would parse the same declarations again outside the prepared state.
            pp.Includes.clear();
            pp.ImplicitPCHInclude = artifact->kind == "pch" ? artifact->payloadPath : artifact->bootstrapPath;

            if (artifact->kind != "pch")
            {
                inv->getHeaderSearchOpts().PrebuiltModuleFiles[artifact->module] = artifact->payloadPath;
            }
        }
        else
        {
            inv = invocation(driverArgs(settings, sourceName, mode, *inputs), fs, diagnostics, &args);

            if (settings.libcudacxx && !artifact)
            {
                // CCCL's freestanding time definitions assume that no system C headers
                // have been included. Reuse the real C types when compiling with Clang.
                // Resolve through the invocation's ordered paths, including caller CCCL paths.
                for (auto const &entry : inv->getHeaderSearchOpts().UserEntries)
                {
                    auto name = std::format("{}/cuda/std/ctime", entry.Path);

                    if (inputs->exists(name))
                    {
                        break;
                    }

                    auto buffer = fs->getBufferForFile(name);

                    if (!buffer)
                    {
                        continue;
                    }

                    auto contents = (*buffer)->getBuffer().str();
                    constexpr std::string_view guard = "#if _CCCL_FREESTANDING()";
                    auto position = contents.find(guard);

                    if (position != std::string::npos)
                    {
                        contents.insert(position + guard.size(), " && !defined(__CLANGRTC_LIBCUDACXX__)");
                        contents.insert(0, "#include <time.h>\n");
                        add(*inputs, name, contents, 0);
                    }

                    break;
                }
            }

            if (settings.preprocessed)
            {
                // Keep the driver's device target, but bypass preprocessing for the
                // expanded source. The driver treats cuda-cpp-output as a host input.
                inv->getFrontendOpts().Inputs.clear();
                inv->getFrontendOpts().Inputs.emplace_back(sourceName,
                                                           clang::InputKind(clang::Language::CUDA).getPreprocessed());
                inv->getPreprocessorOpts().Includes.clear();
            }
        }

        result->measurements.emplace_back("invocation", elapsed(start));
        return inv;
    }

    /**
     * @brief Creates an isolated frontend with recording inputs and memory outputs.
     *
     * Each frontend receives its own module cache so unrelated virtual-file
     * identities cannot reuse stale serialized state. PCH container handlers are
     * owned by each frontend; the diagnostic consumer remains borrowed.
     *
     * @param inv Invocation transferred into the new frontend.
     * @param diag Diagnostic consumer that must outlive the returned frontend.
     * @return Owned compiler instance with no action executed yet.
     */
    std::unique_ptr<clang::CompilerInstance> compiler(std::shared_ptr<clang::CompilerInvocation> inv, Diagnostics &diag)
    {
        auto start = Clock::now();
        CR_DEFER([&] { result->frontendNs += elapsed(start); });
        auto ci = std::make_unique<clang::CompilerInstance>(
            std::move(inv), std::make_shared<clang::PCHContainerOperations>(), clang::createCrossProcessModuleCache());
        ci->setVirtualFileSystem(fs);
        ci->setOutputManager(llvm::makeIntrusiveRefCnt<MemoryOutputs>());
        ci->createDiagnostics(&diag, false);
        return ci;
    }
};

/**
 * @brief Adapts LLVM layout and entry metadata for downstream NVVM compilation.
 *
 * Checks identified struct sizes, alignments, and member offsets before
 * installing the NVVM data layout. Selected definitions are retained and
 * kernels receive NVVM annotations. With explicit entries, other definitions
 * are internalized so later optimization can remove unreachable code. LLVM
 * bookkeeping globals are excluded from the retained-root list.
 *
 * @param m Generated module modified in place.
 * @param result Result containing resolved entries, or an empty list to retain visible definitions.
 * @param device Whether to install NVVM layout and metadata instead of preserving the host ABI.
 * @throws Failure if a checked record ABI changes or a requested definition is missing.
 */
void normalize(llvm::Module &m, clangrtcResult result, bool device)
{
    if (device)
    {
        llvm::DataLayout nvvm("e-p:64:64:64-p6:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-"
                              "i128:128:128-f32:32:32-f64:64:64-f128:128:128-v16:16:16-v32:32:32-v64:"
                              "64:64-v128:128:128-n16:32:64-a:8:8");
        // NVVM expects its own data layout. Reject records whose ABI would change
        // before replacing Clang's layout, rather than silently changing kernel arguments.
        auto const &original = m.getDataLayout();

        for (auto *type : m.getIdentifiedStructTypes())
        {
            if (type->isSized())
            {
                CR_EXPECTS(original.getTypeAllocSize(type) == nvvm.getTypeAllocSize(type) &&
                               original.getABITypeAlign(type) == nvvm.getABITypeAlign(type),
                           "NVVM layout changes struct ABI", ::cr::Failure, CLANGRTC_UNSUPPORTED);
                auto *a = original.getStructLayout(type), *b = nvvm.getStructLayout(type);

                for (unsigned i = 0; i < type->getNumElements(); ++i)
                {
                    CR_EXPECTS(a->getElementOffset(i) == b->getElementOffset(i), "NVVM layout changes member offset",
                               ::cr::Failure, CLANGRTC_UNSUPPORTED);
                }
            }
        }

        m.setDataLayout(nvvm);
        auto &ctx = m.getContext();

        if (!m.getNamedMetadata("nvvmir.version"))
        {
            m.getOrInsertNamedMetadata("nvvmir.version")
                ->addOperand(llvm::MDNode::get(
                    ctx, {llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx), 2)),
                          llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx), 0)),
                          llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx), 3)),
                          llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx), 2))}));
        }
    }

    auto &ctx = m.getContext();
    std::vector<llvm::GlobalValue *> roots;

    if (result->names.empty())
    {
        for (auto &f : m)
        {
            if (!f.isDeclaration() && !f.hasLocalLinkage())
            {
                roots.push_back(&f);
            }
        }

        for (auto &g : m.globals())
        {
            if (!g.isDeclaration() && !g.hasLocalLinkage() && !g.getName().starts_with("llvm."))
            {
                roots.push_back(&g);
            }
        }
    }

    for (auto const *names : {&result->artifactNames, &result->names})
    {
        for (auto const &[expression, name] : *names)
        {
            auto *root = m.getNamedValue(name);
            CR_EXPECTS(root && !root->isDeclaration(), std::format("No definition emitted for {}", expression),
                       ::cr::Failure, CLANGRTC_COMPILATION_FAILED);
            root->setLinkage(llvm::GlobalValue::ExternalLinkage);

            if (auto *object = llvm::dyn_cast<llvm::GlobalObject>(root))
            {
                // Selected entries are strong definitions. Leaving a template COMDAT
                // attached could let linking discard their bodies in favor of a
                // consumer group whose copy was already optimized away.
                object->setComdat(nullptr);
            }

            if (std::find(roots.begin(), roots.end(), root) == roots.end())
            {
                roots.push_back(root);
            }
        }
    }

    // Preserve externally requested entries even when no other emitted function calls them.
    llvm::appendToCompilerUsed(m, roots);

    for (auto *root : roots)
    {
        if (auto *fn = llvm::dyn_cast<llvm::Function>(root);
            fn && fn->getCallingConv() == llvm::CallingConv::PTX_Kernel)
        {
            m.getOrInsertNamedMetadata("nvvm.annotations")
                ->addOperand(llvm::MDNode::get(
                    ctx, {llvm::ValueAsMetadata::get(fn), llvm::MDString::get(ctx, "kernel"),
                          llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx), 1))}));
        }
    }

    if (!result->names.empty())
    {
        std::set<std::string> names;

        for (auto *root : roots)
        {
            names.insert(root->getName().str());
        }

        // Only requested entries stay externally visible; optimization can discard
        // unreferenced helpers while retaining dependencies reachable from those entries.
        llvm::internalizeModule(m, [&](llvm::GlobalValue const &g) {
            return names.contains(g.getName().str()) || g.getName().starts_with("llvm.");
        });
    }
}

/**
 * @brief Runs LLVM's standard module optimization pipeline.
 *
 * @param m Module optimized in place.
 * @param level Validated optimization level from zero through three.
 */
void optimize(llvm::Module &m, int level)
{
    llvm::LoopAnalysisManager loops;
    llvm::FunctionAnalysisManager functions;
    llvm::CGSCCAnalysisManager scc;
    llvm::ModuleAnalysisManager modules;
    llvm::PassBuilder builder;
    builder.registerModuleAnalyses(modules);
    builder.registerCGSCCAnalyses(scc);
    builder.registerFunctionAnalyses(functions);
    builder.registerLoopAnalyses(loops);
    builder.crossRegisterProxies(loops, functions, scc, modules);
    auto pipeline = level ? builder.buildPerModuleDefaultPipeline(level == 1   ? llvm::OptimizationLevel::O1
                                                                  : level == 2 ? llvm::OptimizationLevel::O2
                                                                               : llvm::OptimizationLevel::O3)
                          : builder.buildO0DefaultPipeline(llvm::OptimizationLevel::O0);
    pipeline.run(m, modules);
}

/**
 * @brief Compiles preparation slots and their implementation dependencies to NVVM bitcode.
 *
 * A standalone bitcode section preserves code that a BMI can omit in favor of a
 * companion object. Consumers link this section instead of resolving prepared
 * expressions again against imported declarations.
 *
 * @param run Preparation inputs and result receiving resolved symbols and timings.
 * @param diag Diagnostic consumer that remains alive throughout frontend execution.
 * @param mode Artifact format, used to preserve named-module frontend settings.
 * @return Owned NVVM-compatible LLVM bitcode containing the prepared entries.
 * @throws Failure If expression resolution, code generation, or IR verification fails.
 */
std::string prepare_entries(Run &run, Diagnostics &diag, std::string const &mode)
{
    // A BMI can omit private bodies supplied by its usual companion object.
    // Store the selected implementation and its dependencies as IR instead;
    // consumers need not reconstruct that code from the serialized AST.
    auto entryInvocation = run.get(mode == "bmi" || mode == "reduced-bmi" ? mode : "compile");
    entryInvocation->getFrontendOpts().ProgramAction = clang::frontend::EmitLLVMOnly;
    auto semantic = run.compiler(std::move(entryInvocation), diag);
    Emit entries(run.result, artifact_expression_prefix, run.settings.requireKernel);
    auto resolution = Clock::now();
    bool compiled{};

    {
        CR_DEFER([&] { run.result->frontendNs += elapsed(resolution); });
        compiled = semantic->ExecuteAction(entries);
    }

    CR_EXPECTS(
        compiled && !semantic->getDiagnostics().hasErrorOccurred(), "Artifact expression resolution failed", Failure,
        run.result->status == CLANGRTC_INVALID_EXPRESSION ? CLANGRTC_INVALID_EXPRESSION : CLANGRTC_COMPILATION_FAILED);
    auto module = entries.takeModule();
    CR_EXPECTS(static_cast<bool>(module), "Compiler produced no prepared LLVM module", Failure,
               CLANGRTC_INTERNAL_ERROR);
    normalize(*module, run.result, run.settings.device);
    optimize(*module, run.settings.optimization);
    std::string errors;
    llvm::raw_string_ostream diagnostics(errors);
    CR_EXPECTS(!llvm::verifyModule(*module, &diagnostics), std::format("Invalid prepared NVVM IR: {}", errors), Failure,
               CLANGRTC_COMPILATION_FAILED);
    std::string ir;
    llvm::raw_string_ostream output(ir);
    llvm::WriteBitcodeToFile(*module, output);
    run.result->measurements.emplace_back("prepared-entries", elapsed(resolution));
    return ir;
}

/**
 * @brief Links an artifact's prepared implementations into the consumer module.
 *
 * Prepared definitions take precedence over copies emitted while compiling the
 * consumer. The combined module is subsequently normalized and optimized once.
 *
 * @param module Consumer IR modified in place.
 * @param bytes Borrowed prepared LLVM bitcode; no terminating NUL is required.
 * @throws Failure If the bitcode cannot be read or linked.
 */
void link_entries(llvm::Module &module, std::span<char const> bytes)
{
    auto prepared = llvm::parseBitcodeFile(
        llvm::MemoryBufferRef(llvm::StringRef(bytes.data(), bytes.size()), "prepared entries"), module.getContext());

    if (!prepared)
    {
        CR_FAIL(std::format("Invalid prepared LLVM bitcode: {}", llvm::toString(prepared.takeError())), Failure,
                CLANGRTC_INVALID_ARTIFACT);
    }

    // Match the consumer's original layout while linking. normalize() checks the
    // combined records before installing the NVVM layout.
    (*prepared)->setDataLayout(module.getDataLayout());
    auto &context = module.getContext();
    std::string log;
    auto handler = std::make_unique<llvm::DiagnosticHandler>(&log);
    handler->DiagHandlerCallback = [](llvm::DiagnosticInfo const *diagnostic, void *data) {
        llvm::raw_string_ostream output(*static_cast<std::string *>(data));
        llvm::DiagnosticPrinterRawOStream printer(output);
        diagnostic->print(printer);
        output << '\n';
    };

    auto previous = context.getDiagnosticHandler();
    CR_DEFER([&] { context.setDiagnosticHandler(std::move(previous)); });
    context.setDiagnosticHandler(std::move(handler));
    CR_EXPECTS(!llvm::Linker::linkModules(module, std::move(*prepared), llvm::Linker::Flags::OverrideFromSrc),
               std::format("Could not link prepared entry definitions: {}", log), Failure, CLANGRTC_LINK_FAILED);
}

/**
 * @brief Owns an optional thread-local LLVM profiler for one operation.
 *
 * An already active profiler is rejected rather than replacing an embedding
 * application's state. Destruction captures trace JSON in the result log and
 * releases only the profiler initialized by this guard.
 */
struct Trace
{
    /**
     * @brief Borrowed result receiving the completed time-trace JSON.
     */
    clangrtcResult result;

    /**
     * @brief Whether this guard owns an initialized profiler.
     */
    bool active;

    /**
     * @brief Starts profiling when requested and no profiler is active.
     *
     * @param enabled Whether this operation requested an LLVM time trace.
     * @param result Result that must outlive this guard.
     * @throws Failure if profiling was requested while another profiler is active.
     */
    Trace(bool enabled, clangrtcResult result) : result(result), active(enabled)
    {
        CR_EXPECTS(!enabled || !llvm::timeTraceProfilerEnabled(), "A time profiler is already active on this thread",
                   ::cr::Failure, CLANGRTC_UNSUPPORTED);

        if (active)
        {
            llvm::timeTraceProfilerInitialize(0, "clangrtc");
        }
    }

    /**
     * @brief Serializes an owned trace into the log and releases its profiler.
     *
     * Does nothing when profiling is disabled. Captured JSON is delimited by
     * CLANGRTC_TIME_TRACE_BEGIN and CLANGRTC_TIME_TRACE_END markers.
     */
    ~Trace()
    {
        if (!active)
        {
            return;
        }

        llvm::SmallString<0> json;
        llvm::raw_svector_ostream output(json);
        llvm::timeTraceProfilerWrite(output);
        llvm::timeTraceProfilerCleanup();
        result->log.append("\nCLANGRTC_TIME_TRACE_BEGIN\n");
        result->log.append(json.data(), json.size());
        result->log.append("\nCLANGRTC_TIME_TRACE_END\n");
    }
};

/**
 * @brief Normalize, optimize, serialize, and finalize one independently owned frontend module.
 * @param module Module whose LLVM context remains alive until this function returns.
 * @param settings Effective output, optimization, target, and backend options.
 * @param artifact Optional prepared implementations borrowed for this call.
 * @param result Current result receiving code, metadata, and stage timings.
 */
void finish_module(std::unique_ptr<llvm::Module> module, Settings const &settings, clangrtcArtifact artifact,
                   clangrtcResult result)
{
    auto begin = Clock::now();

    if (artifact && !artifact->ir.empty())
    {
        link_entries(*module, artifact->ir);
    }

    // Retain selected entries before optimization. Device modules also need
    // NVVM ABI normalization; host modules retain Clang's native layout.
    normalize(*module, result, settings.device);
    optimize(*module, settings.optimization);
    std::string errors;
    llvm::raw_string_ostream err(errors);
    CR_EXPECTS(!llvm::verifyModule(*module, &err), std::format("Invalid generated LLVM IR: {}", errors), ::cr::Failure,
               CLANGRTC_COMPILATION_FAILED);

    for (auto const &[expression, name] : result->names)
    {
        CR_EXPECTS(module->getNamedValue(name), std::format("Selected entry was lost: {}", expression), ::cr::Failure,
                   CLANGRTC_COMPILATION_FAILED);
    }

    result->measurements.emplace_back("llvm", elapsed(begin));
    begin = Clock::now();

    if (settings.format != "object")
    {
        llvm::raw_string_ostream os(result->output);

        if (settings.format == "nvvm-ir-text" || settings.format == "llvm-ir-text")
        {
            module->print(os, nullptr);
        }
        else
        {
            llvm::WriteBitcodeToFile(*module, os);
        }

        result->measurements.emplace_back("serialization", elapsed(begin));
    }

    if (settings.format == "object")
    {
        begin = Clock::now();
        static std::once_flag nativeTarget;
        std::call_once(nativeTarget, [] {
            CR_EXPECTS(!llvm::InitializeNativeTarget() && !llvm::InitializeNativeTargetAsmPrinter(),
                       "Native LLVM backend is unavailable", Failure, CLANGRTC_TOOLCHAIN_ERROR);
        });
        std::string error;
        auto *target = llvm::TargetRegistry::lookupTarget(module->getTargetTriple(), error);
        CR_EXPECTS(target, error, Failure, CLANGRTC_TOOLCHAIN_ERROR);
        std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(
            module->getTargetTriple(), "generic", "", llvm::TargetOptions{}, llvm::Reloc::PIC_));
        CR_EXPECTS(machine, "Cannot create native target machine", Failure, CLANGRTC_TOOLCHAIN_ERROR);
        llvm::SmallVector<char, 0> bytes;
        llvm::raw_svector_ostream object(bytes);
        llvm::legacy::PassManager passes;
        CR_EXPECTS(!machine->addPassesToEmitFile(passes, object, nullptr, llvm::CodeGenFileType::ObjectFile),
                   "Native target cannot emit object files", Failure, CLANGRTC_UNSUPPORTED);
        passes.run(*module);
        result->output.assign(bytes.begin(), bytes.end());
        result->measurements.emplace_back("codegen", elapsed(begin));
    }

    if (settings.format == "lto-ir" || settings.format == "ptx-text")
    {
        begin = Clock::now();
        nvvmProgram program{};
        CR_DEFER([&] {
            if (program)
            {
                nvvm().nvvmDestroyProgram(&program);
            }
        });
        CR_CHECK_NVVM(nvvm().nvvmCreateProgram(&program), program);
        CR_CHECK_NVVM(nvvm().nvvmAddModuleToProgram(program, result->output.data(), result->output.size(), "clangrtc"),
                      program);
        auto architecture = std::format("-arch=compute_{}", settings.target.substr(3));
        std::vector<char const *> options{architecture.c_str(), "-nvvm-latest"};

        if (settings.format == "lto-ir")
        {
            options.push_back("-gen-lto");
        }

        for (auto const &option : settings.nvvmOptions)
        {
            options.push_back(option.c_str());
        }

        CR_CHECK_NVVM(nvvm().nvvmCompileProgram(program, static_cast<int>(options.size()), options.data()), program);
        size_t size{};
        CR_CHECK_NVVM(nvvm().nvvmGetCompiledResultSize(program, &size), program);
        std::string bytes(size, '\0');
        CR_CHECK_NVVM(nvvm().nvvmGetCompiledResult(program, bytes.data()), program);

        if (settings.format == "ptx-text" && !bytes.empty() && bytes.back() == '\0')
        {
            bytes.pop_back();
        }

        result->output = std::move(bytes);
        result->measurements.emplace_back("nvvm", elapsed(begin));
    }

    result->format = settings.format;
}

/**
 * @brief Executes the common prepare/compile path and translates operation failures.
 *
 * Each call owns its frontend and diagnostic configuration. Successful preparation
 * serializes dependencies, compiler state, and optional entry bitcode.
 * Compilation resolves new entries, links prepared implementations, optimizes and
 * verifies the combined LLVM module, and serializes the result.
 * Source and option failures retain an owned diagnostic result when possible.
 *
 * @param request Source and associated request inputs.
 * @param opts Call-specific option array, or NULL when n is zero.
 * @param n Number of call-specific options.
 * @param callback Optional synchronous diagnostic callback for this call.
 * @param user Caller context borrowed while delivering diagnostics.
 * @param output Destination for an owned result, cleared before request validation.
 * @param preparing Whether to prepare an artifact instead of emitting device IR.
 * @return Operation status; destroy every non-NULL result even when it indicates failure.
 */
clangrtcStatus operate(clangrtcRequest const *request, clangrtcString const *opts, size_t n,
                       clangrtcDiagnosticCallback callback, void *user, clangrtcResult *output, bool preparing)
{
    if (!output)
    {
        return CLANGRTC_INVALID_ARGUMENT;
    }

    *output = nullptr;

    if (!request)
    {
        return CLANGRTC_INVALID_ARGUMENT;
    }

    auto result = std::unique_ptr<clangrtcResultImpl>(new (std::nothrow) clangrtcResultImpl);

    if (!result)
    {
        return CLANGRTC_OUT_OF_MEMORY;
    }

    auto start = Clock::now();
    Diagnostics diag(result.get(), callback, user);

    try
    {
        Run run(result.get(), diag, *request, opts, n, preparing);
        auto mode = preparing ? run.settings.artifact : std::string("compile");

        Trace trace(run.settings.trace, result.get());

        auto inv = run.get(mode);

        if (preparing && mode == "pch")
        {
            inv->getLangOpts().PCHInstantiateTemplates = true;
        }

        auto ci = run.compiler(inv, diag);

        if (preparing)
        {
            clangrtcArtifactImpl a;
            a.kind = mode;
            a.target = run.settings.device ? run.settings.target : llvm::sys::getDefaultTargetTriple();
            a.options = run.options;
            a.source = run.sourceName;
            a.args = run.args;
            std::string moduleBytes;
            auto action = mode == "pch" ? std::unique_ptr<clang::FrontendAction>(new clang::GeneratePCHAction)
                          : mode == "bmi" || mode == "reduced-bmi"
                              ? std::unique_ptr<clang::FrontendAction>(
                                    new MemoryModuleAction(mode == "reduced-bmi", moduleBytes, a.module))
                          : mode == "preprocessed-cuda"
                              ? std::unique_ptr<clang::FrontendAction>(new clang::PrintPreprocessedAction)
                              : std::unique_ptr<clang::FrontendAction>(new clang::PreprocessOnlyAction);
            auto begin = Clock::now();
            bool ok{};

            {
                CR_DEFER([&] { result->frontendNs += elapsed(begin); });
                ok = ci->ExecuteAction(*action);
            }
            result->measurements.emplace_back("preparation", elapsed(begin));
            CR_EXPECTS(ok && !ci->getDiagnostics().hasErrorOccurred(), "Artifact preparation failed", ::cr::Failure,
                       result->status == CLANGRTC_INVALID_EXPRESSION ? CLANGRTC_INVALID_EXPRESSION
                                                                     : CLANGRTC_COMPILATION_FAILED);
            a.payload = mode == "cuda"                           ? run.source
                        : mode == "bmi" || mode == "reduced-bmi" ? moduleBytes
                        : mode == "preprocessed-cuda"            ? captured(*ci)
                                                                 : captured(*ci);

            if (!result->names.empty())
            {
                a.ir = prepare_entries(run, diag, mode);
            }

            a.names = result->names;
            a.cudaNames = result->cudaNames;
            a.files = run.files;

            if (mode == "bmi" || mode == "reduced-bmi")
            {
                // A BMI provides module declarations, while the bootstrap PCH restores
                // the device-header environment needed to parse its importing consumer.
                auto empty = "/__clangrtc__/bootstrap.cuh";
                add(*run.inputs, empty, "#pragma once\n", 0);
                std::vector<std::string> saved;
                auto bootstrap = invocation(driverArgs(run.settings, empty, "pch", *run.inputs), run.fs, diag, &saved);
                // Finish header-only pending instantiations once during AOT preparation, rather
                // than repeating standard-library template work in every importing consumer.
                bootstrap->getLangOpts().PCHInstantiateTemplates = true;
                auto boot = run.compiler(bootstrap, diag);
                clang::GeneratePCHAction pch;
                auto begin = Clock::now();
                bool ok{};

                {
                    CR_DEFER([&] { result->frontendNs += elapsed(begin); });
                    ok = boot->ExecuteAction(pch);
                }

                CR_EXPECTS(ok, "Bootstrap PCH preparation failed", ::cr::Failure, CLANGRTC_COMPILATION_FAILED);
                a.bootstrap = captured(*boot);
                a.files = run.files;
            }

            CR_EXPECTS(!a.payload.empty(), "Compiler produced no artifact", ::cr::Failure, CLANGRTC_COMPILATION_FAILED);
            auto serialization = Clock::now();
            result->output = serialize(a);
            result->measurements.emplace_back("serialization", elapsed(serialization));
            result->format = "artifact";
        }
        else
        {
            Emit action(result.get(), compilation_expression_prefix, run.settings.requireKernel);
            auto begin = Clock::now();
            bool ok{};

            {
                CR_DEFER([&] { result->frontendNs += elapsed(begin); });
                ok = ci->ExecuteAction(action);
            }

            CR_EXPECTS(ok && !ci->getDiagnostics().hasErrorOccurred(), "Compilation failed", ::cr::Failure,
                       result->status == CLANGRTC_INVALID_EXPRESSION ? CLANGRTC_INVALID_EXPRESSION
                                                                     : CLANGRTC_COMPILATION_FAILED);
            auto module = action.takeModule();
            CR_EXPECTS(static_cast<bool>(module), "Compiler produced no LLVM module", ::cr::Failure,
                       CLANGRTC_INTERNAL_ERROR);
            finish_module(std::move(module), run.settings, run.artifact, result.get());
        }
    }
    catch (Failure const &e)
    {
        result->status = e.status;
        diag.message(e.what(), e.stage, CLANGRTC_DIAGNOSTIC_ERROR, {}, 0, 0);
        result->output.clear();
        result->format.clear();
    }
    catch (std::bad_alloc const &)
    {
        result->status = CLANGRTC_OUT_OF_MEMORY;
        result->output.clear();
    }
    catch (std::exception const &e)
    {
        result->status = CLANGRTC_INTERNAL_ERROR;
        diag.message(e.what(), "frontend", CLANGRTC_DIAGNOSTIC_ERROR, {}, 0, 0);
        result->output.clear();
    }

    result->total = elapsed(start);
    auto status = result->status;
    // Failure results also transfer to the caller so diagnostics remain available
    // after the frontend and its temporary buffers have been destroyed.
    *output = result.release();
    return status;
}

} // namespace cr

extern "C"
{
    clangrtcStatus clangrtcInit(clangrtcDiagnosticCallback callback, void *userData)
    {
        try
        {
            (void)cr::nvvm();
            return CLANGRTC_SUCCESS;
        }
        catch (cr::Failure const &failure)
        {
            if (callback)
            {
                clangrtcDiagnostic diagnostic{
                    CLANGRTC_DIAGNOSTIC_ERROR, cr::to_view(failure.stage), cr::to_view(failure.what()), {}, 0, 0};

                try
                {
                    callback(&diagnostic, userData);
                }
                catch (...)
                {
                }
            }

            return failure.status;
        }
        catch (std::bad_alloc const &)
        {
            return CLANGRTC_OUT_OF_MEMORY;
        }
        catch (...)
        {
            return CLANGRTC_INTERNAL_ERROR;
        }
    }

    /**
     * @brief Returns compiler identity and capabilities backed by static storage.
     *
     * This query allocates no compiler handle. All returned views remain valid
     * for the lifetime of the library.
     */
    clangrtcStatus clangrtcGetInfo(clangrtcCompilerInfo *info)
    {
        if (!info)
        {
            return CLANGRTC_INVALID_ARGUMENT;
        }

        static constexpr std::array capabilities{cr::to_view("artifact:pch"),
                                                 cr::to_view("artifact:bmi"),
                                                 cr::to_view("artifact:reduced-bmi"),
                                                 cr::to_view("artifact:cuda"),
                                                 cr::to_view("artifact:preprocessed-cuda"),
                                                 cr::to_view("output:nvvm-ir-bitcode"),
                                                 cr::to_view("output:nvvm-ir-text"),
                                                 cr::to_view("output:lto-ir"),
                                                 cr::to_view("output:ptx-text"),
                                                 cr::to_view("output:llvm-ir-bitcode"),
                                                 cr::to_view("output:llvm-ir-text"),
                                                 cr::to_view("output:object"),
                                                 cr::to_view("host-compilation"),
                                                 cr::to_view("options:clang"),
                                                 cr::to_view("options:nvvm"),
                                                 cr::to_view("version-queries"),
                                                 cr::to_view("name-expressions"),
                                                 cr::to_view("memory-artifacts"),
                                                 cr::to_view("profile:nvrtc"),
                                                 cr::to_view("profile:hosted")};

        *info = {cr::to_view(cr::libraryVersion), cr::to_view(CLANG_VERSION_STRING), cr::to_view(LLVM_VERSION_STRING),
                 capabilities.data(), capabilities.size()};
        return CLANGRTC_SUCCESS;
    }

    clangrtcStatus clangrtcGetVersion(clangrtcString component, clangrtcString *version)
    {
        if (!version)
        {
            return CLANGRTC_INVALID_ARGUMENT;
        }

        *version = {};

        if (!component.data || !component.size)
        {
            return CLANGRTC_INVALID_ARGUMENT;
        }

        auto name = std::string_view(component.data, component.size);

        if (name == "clang")
        {
            *version = cr::to_view(CLANG_VERSION_STRING);
            return CLANGRTC_SUCCESS;
        }

        if (name == "llvm")
        {
            *version = cr::to_view(LLVM_VERSION_STRING);
            return CLANGRTC_SUCCESS;
        }

        if (name != "nvvm")
        {
            return CLANGRTC_INVALID_ARGUMENT;
        }

        try
        {
            /** @brief Cached libNVVM version query and its API status. */
            struct Version
            {
                /** @brief Status returned by the version query. */
                clangrtcStatus status;

                /** @brief Stable version text borrowed by successful API calls. */
                std::string text;
            };

            static Version const value = [] {
                int major{}, minor{};

                if (cr::nvvm().nvvmVersion(&major, &minor) != NVVM_SUCCESS)
                {
                    return Version{CLANGRTC_INTERNAL_ERROR, {}};
                }

                return Version{CLANGRTC_SUCCESS, std::format("{}.{}", major, minor)};
            }();

            if (value.status == CLANGRTC_SUCCESS)
            {
                *version = cr::to_view(value.text);
            }

            return value.status;
        }
        catch (cr::Failure const &failure)
        {
            return failure.status;
        }
        catch (std::bad_alloc const &)
        {
            return CLANGRTC_OUT_OF_MEMORY;
        }
        catch (...)
        {
            return CLANGRTC_INTERNAL_ERROR;
        }
    }

    /**
     * @brief Routes explicit artifact preparation through the common operation path.
     *
     * The shared path owns request validation, per-call diagnostics, dependency
     * packaging, and failure-to-status conversion.
     */
    clangrtcStatus clangrtcPrepare(clangrtcRequest const *request, clangrtcString const *options, size_t optionCount,
                                   clangrtcDiagnosticCallback callback, void *userData, clangrtcResult *result)
    {
        return cr::operate(request, options, optionCount, callback, userData, result, true);
    }

    /**
     * @brief Routes device IR compilation through the common operation path.
     *
     * The shared path owns prepared-state validation, symbol resolution,
     * IR verification and serialization, diagnostics, and timing.
     */
    clangrtcStatus clangrtcCompile(clangrtcRequest const *request, clangrtcString const *options, size_t optionCount,
                                   clangrtcDiagnosticCallback callback, void *userData, clangrtcResult *result)
    {
        return cr::operate(request, options, optionCount, callback, userData, result, false);
    }

    /**
     * @brief Copies and validates a package before publishing its owned artifact handle.
     *
     * Parsing failures are sent to this call's diagnostic callback because
     * artifact loading returns no operation result log.
     */
    clangrtcStatus clangrtcLoadArtifact(clangrtcString package, clangrtcDiagnosticCallback callback, void *userData,
                                        clangrtcArtifact *artifact)
    {
        if (!artifact)
        {
            return CLANGRTC_INVALID_ARGUMENT;
        }

        *artifact = nullptr;

        try
        {
            auto a = cr::deserialize(std::span<char const>{package.data, package.size});

            *artifact = a.release();
            return CLANGRTC_SUCCESS;
        }
        catch (cr::Failure const &e)
        {
            clangrtcResultImpl result;
            cr::Diagnostics diagnostics(&result, callback, userData);
            diagnostics.message(e.what(), "artifact-loading", CLANGRTC_DIAGNOSTIC_ERROR, {}, 0, 0);
            return e.status;
        }
        catch (std::bad_alloc const &)
        {
            return CLANGRTC_OUT_OF_MEMORY;
        }
        catch (...)
        {
            return CLANGRTC_INTERNAL_ERROR;
        }
    }

    /**
     * @brief Returns stable metadata views borrowed from a loaded artifact.
     *
     * Inspection does not instantiate compiler state or copy package contents.
     * The artifact must remain alive while its metadata views are in use.
     */
    clangrtcStatus clangrtcGetArtifactInfo(clangrtcArtifact artifact, clangrtcArtifactInfo *info)
    {
        if (!artifact || !info)
        {
            return CLANGRTC_INVALID_ARGUMENT;
        }

        *info = {cr::to_view(artifact->kind),
                 cr::to_view(artifact->compiler),
                 cr::to_view(artifact->target),
                 cr::to_view(artifact->module),
                 artifact->size,
                 cr::to_view(artifact->contentHash),
                 artifact->dependencyViews.data(),
                 artifact->dependencyViews.size(),
                 artifact->optionViews.data(),
                 artifact->optionViews.size(),
                 artifact->symbols.data(),
                 artifact->symbols.size()};
        return CLANGRTC_SUCCESS;
    }

    /**
     * @brief Releases an artifact and its packaged memory filesystem.
     *
     * Deleting a NULL handle is a no-op. All requests borrowing the artifact
     * must have returned before it is destroyed.
     */
    void clangrtcDestroyArtifact(clangrtcArtifact artifact)
    {
        delete artifact;
    }

    /**
     * @brief Materializes stable public views over owned result storage.
     *
     * Symbol and timing arrays are built once, preserving earlier views
     * on repeated retrieval. The returned API status describes retrieval;
     * view.status describes the original compilation or preparation operation.
     */
    clangrtcStatus clangrtcGetResult(clangrtcResult result, clangrtcResultView *view)
    {
        if (!result || !view)
        {
            return CLANGRTC_INVALID_ARGUMENT;
        }

        try
        {
            if (!result->viewsReady)
            {
                result->symbols.reserve(result->names.size());
                result->timings.reserve(result->measurements.size() + 1);

                for (auto const &[e, n] : result->names)
                {
                    result->symbols.push_back({cr::to_view(e), cr::to_view(n), cr::to_view(result->cudaNames[n])});
                }

                for (auto const &[s, n] : result->measurements)
                {
                    result->timings.push_back({cr::to_view(s), n});
                }

                if (result->frontendNs)
                {
                    result->timings.push_back({cr::to_view("frontend"), result->frontendNs});
                }

                result->viewsReady = true;
            }

            *view = {result->status,           cr::to_view(result->format), cr::to_view(result->output),
                     cr::to_view(result->log), result->symbols.data(),      result->symbols.size(),
                     result->timings.data(),   result->timings.size(),      result->total,
                     result->frontendNs};
            return CLANGRTC_SUCCESS;
        }
        catch (std::bad_alloc const &)
        {
            return CLANGRTC_OUT_OF_MEMORY;
        }
    }

    /**
     * @brief Releases an operation result and invalidates every borrowed result view.
     *
     * Deleting a NULL handle is a no-op. Results own their storage
     * until this handle is destroyed, independently of other API calls.
     */
    void clangrtcDestroyResult(clangrtcResult result)
    {
        delete result;
    }

    /**
     * @brief Returns a static text description for a public status value.
     *
     * Unknown values return the static fallback description. The view length
     * excludes the NUL terminator, which callers do not need to inspect.
     */
    clangrtcString clangrtcStatusString(clangrtcStatus status)
    {
        switch (status)
        {
        case CLANGRTC_SUCCESS:
            return cr::to_view("success");
        case CLANGRTC_INVALID_ARGUMENT:
            return cr::to_view("invalid argument");
        case CLANGRTC_INVALID_OPTION:
            return cr::to_view("invalid option");
        case CLANGRTC_INCOMPATIBLE_ARTIFACT:
            return cr::to_view("incompatible artifact");
        case CLANGRTC_UNSUPPORTED:
            return cr::to_view("unsupported");
        case CLANGRTC_COMPILATION_FAILED:
            return cr::to_view("compilation failed");
        case CLANGRTC_OUT_OF_MEMORY:
            return cr::to_view("out of memory");
        case CLANGRTC_INTERNAL_ERROR:
            return cr::to_view("internal error");
        case CLANGRTC_INVALID_ARTIFACT:
            return cr::to_view("invalid artifact");
        case CLANGRTC_TOOLCHAIN_ERROR:
            return cr::to_view("toolchain error");
        case CLANGRTC_INVALID_EXPRESSION:
            return cr::to_view("invalid expression");
        case CLANGRTC_LINK_FAILED:
            return cr::to_view("link failed");
        }

        return cr::to_view("unknown status");
    }
}
