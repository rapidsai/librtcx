/* SPDX-License-Identifier: Apache-2.0 */
#ifndef CLANGRTC_H
#define CLANGRTC_H

/**
 * @file clangrtc.h
 * @brief C interface for in-process CUDA device and native host compilation with Clang.
 *
 * Compilation produces NVVM-compatible LLVM bitcode, textual LLVM IR, NVIDIA
 * LTO IR (--lto-ir), or PTX text (--ptx-text) using libNVVM.
 * Host-only C/C++ additionally supports native LLVM IR and relocatable objects.
 * Artifact preparation packages CUDA source, preprocessed CUDA, or serialized
 * Clang state for later compilation from memory. Configuration is supplied as
 * arrays of option strings; no input string requires a terminating NUL byte.
 *
 * Inputs are borrowed for the duration of each call unless stated otherwise.
 * Functions returning handles transfer ownership to the caller. Destroy each
 * handle with its matching destroy function; never free borrowed storage.
 * Passing a destroyed handle or destroying a handle while it is in use is
 * invalid. Preparation, compilation, and loading use call-local state and may
 * run concurrently. Serialize accesses to the same result handle. Immutable
 * artifacts may be shared by concurrent calls until all users have returned.
 */

#include <stddef.h>
#include <stdint.h>

/** @cond CLANGRTC_IMPLEMENTATION */
#if defined(_WIN32)
#if defined(CLANGRTC_BUILDING_LIBRARY)
#define CLANGRTC_API __declspec(dllexport)
#else
#define CLANGRTC_API __declspec(dllimport)
#endif
#else
#define CLANGRTC_API __attribute__((visibility("default")))
#endif
/** @endcond */

#ifdef __cplusplus
extern "C"
{
#endif

    /**
     * @brief Non-owning view of text or binary data.
     *
     * No terminating NUL byte is required or included in @ref size. Input views
     * must describe valid byte ranges. Copying a view does not scan its contents.
     * A zero-initialized view represents empty data. Returned views follow the
     * lifetime of their artifact or result, or static library storage as documented below.
     */
    typedef struct
    {
        /**
         * @brief First byte; may be NULL only when size is zero.
         */
        const char *data;

        /**
         * @brief Number of bytes accessible through data.
         */
        size_t size;
    } clangrtcString;

    /**
     * @brief Owned, immutable artifact; release with clangrtcDestroyArtifact().
     */
    typedef struct clangrtcArtifactImpl *clangrtcArtifact;

    /**
     * @brief Owned operation result; release with clangrtcDestroyResult().
     */
    typedef struct clangrtcResultImpl *clangrtcResult;

    /**
     * @brief Outcome of an API call or compilation operation.
     *
     * Numeric values are fixed. Value 8 is reserved.
     */
    typedef enum
    {
        /**
         * @brief The operation succeeded.
         */
        CLANGRTC_SUCCESS = 0,

        /**
         * @brief An API argument violates its contract.
         */
        CLANGRTC_INVALID_ARGUMENT = 1,

        /**
         * @brief Invalid option or value.
         */
        CLANGRTC_INVALID_OPTION = 2,

        /**
         * @brief Incompatible prepared state.
         */
        CLANGRTC_INCOMPATIBLE_ARTIFACT = 3,

        /**
         * @brief Unsupported behavior.
         */
        CLANGRTC_UNSUPPORTED = 4,

        /**
         * @brief Compilation or preparation failed.
         */
        CLANGRTC_COMPILATION_FAILED = 5,

        /**
         * @brief A host allocation failed.
         */
        CLANGRTC_OUT_OF_MEMORY = 6,

        /**
         * @brief An internal operation failed.
         */
        CLANGRTC_INTERNAL_ERROR = 7,

        /**
         * @brief The artifact is malformed or corrupted; regenerate or replace it.
         */
        CLANGRTC_INVALID_ARTIFACT = 9,

        /**
         * @brief Required CUDA or Clang files are missing, unreadable, or misconfigured.
         */
        CLANGRTC_TOOLCHAIN_ERROR = 10,

        /**
         * @brief An expression slot cannot identify a supported definition.
         */
        CLANGRTC_INVALID_EXPRESSION = 11,

        /**
         * @brief Prepared implementations could not be linked with the consumer.
         */
        CLANGRTC_LINK_FAILED = 12
    } clangrtcStatus;

    /**
     * @brief Severity of a diagnostic delivered to a callback.
     */
    typedef enum
    {
        /**
         * @brief Context or an informational note.
         */
        CLANGRTC_DIAGNOSTIC_NOTE = 0,

        /**
         * @brief Warning; compilation may succeed.
         */
        CLANGRTC_DIAGNOSTIC_WARNING = 1,

        /**
         * @brief Error; the operation cannot complete.
         */
        CLANGRTC_DIAGNOSTIC_ERROR = 2
    } clangrtcDiagnosticLevel;

    /**
     * @brief A diagnostic borrowed for the duration of its callback.
     *
     * Copy any fields that need to survive the callback. Locations are one-based
     * when available. A missing location has an empty file and zero line/column.
     */
    typedef struct
    {
        /**
         * @brief Diagnostic severity.
         */
        clangrtcDiagnosticLevel level;

        /**
         * @brief Producing stage, such as "frontend".
         */
        clangrtcString stage;

        /**
         * @brief Human-readable diagnostic text.
         */
        clangrtcString message;

        /**
         * @brief Source filename, or an empty view.
         */
        clangrtcString file;

        /**
         * @brief Source line, or zero if unavailable.
         */
        unsigned int line;

        /**
         * @brief Source column, or zero if unavailable.
         */
        unsigned int column;
    } clangrtcDiagnostic;

    /**
     * @brief Receives diagnostics synchronously on the thread running the
     * operation.
     *
     * The callback must not throw exceptions. Its function and context apply only
     * to this operation; nested calls specify their own diagnostic configuration.
     *
     * @param diagnostic Borrowed diagnostic, valid until the callback returns.
     * @param userData Caller-owned pointer supplied with this operation.
     */
    typedef void (*clangrtcDiagnosticCallback)(const clangrtcDiagnostic *diagnostic, void *userData);

    /**
     * @brief An in-memory source file or header, borrowed for the current call.
     */
    typedef struct
    {
        /**
         * @brief Virtual path for includes and diagnostics.
         */
        clangrtcString path;

        /**
         * @brief File contents, read using the explicit byte count.
         */
        clangrtcString data;
    } clangrtcFile;

    /**
     * @brief Inputs to preparation or compilation.
     *
     * Zero-initialize this structure before assigning fields. Paths identify files
     * in the compilation's virtual filesystem; relative paths resolve against the
     * process working directory. Additional headers may satisfy includes without
     * existing on disk. Compilation cannot replace a packaged dependency with
     * different contents.
     */
    typedef struct
    {
        /**
         * @brief Source text and its virtual filename.
         *
         * An empty path selects an internal filename. An empty source is allowed
         * when preparing an empty prefix or
         * compiling the source stored in a CUDA/preprocessed-CUDA artifact.
         */
        clangrtcFile source;

        /**
         * @brief Optional prepared artifact for compilation.
         *
         * Must be NULL for preparation. The artifact is borrowed and must remain
         * alive until the call returns.
         */
        clangrtcArtifact artifact;

        /**
         * @brief Virtual headers, or NULL.
         */
        const clangrtcFile *headers;

        /**
         * @brief Number of entries in headers.
         */
        size_t headerCount;

        /**
         * @brief Entry expressions resolved by Clang during preparation or compilation.
         *
         * Each expression, such as "&kernel<int>", must identify a device function,
         * kernel, or device variable. Host C++ mode accepts host functions and variables;
         * C mode accepts plain function/variable identifiers and returns their C ABI names
         * unchanged. Symbols are returned in request order.
         * Preparation retains these entries in the artifact and records their slots
         * in clangrtcArtifactInfo::symbols. Compilation retains both the artifact's
         * entries and its own requested entries, but returns only its own slots in
         * clangrtcResultView::symbols. Slot numbering is independent for each
         * request. With no compilation expressions, externally visible definitions
         * are also retained.
         */
        const clangrtcString *expressions;

        /**
         * @brief Number of entries in expressions.
         */
        size_t expressionCount;
    } clangrtcRequest;

    /**
     * @brief A requested expression and its resolved symbol.
     */
    typedef struct
    {
        /**
         * @brief Requested expression.
         */
        clangrtcString expression;

        /**
         * @brief Resolved symbol; may be empty on failure.
         */
        clangrtcString name;

        /**
         * @brief Qualified CUDA kernel source name with template arguments; empty for other declarations.
         */
        clangrtcString cudaName;
    } clangrtcSymbol;

    /**
     * @brief Host wall-clock duration of a named operation stage.
     */
    typedef struct
    {
        /**
         * @brief Stage name, e.g. "frontend".
         */
        clangrtcString stage;

        /**
         * @brief Duration in nanoseconds.
         */
        uint64_t elapsedNs;
    } clangrtcTiming;

    /**
     * @brief Borrowed view of an operation's output, diagnostics, and measurements.
     *
     * All referenced storage belongs to the result and remains valid until
     * clangrtcDestroyResult(), independently of later API calls. Strings
     * need not be NUL-terminated. Inspect @ref status before using output; failed
     * operations produce no usable compilation output.
     */
    typedef struct
    {
        /**
         * @brief Operation status.
         */
        clangrtcStatus status;

        /**
         * @brief "artifact", "nvvm-ir-bitcode", "nvvm-ir-text", "lto-ir", "ptx-text",
         * "llvm-ir-bitcode", "llvm-ir-text", or "object" on success.
         */
        clangrtcString format;

        /**
         * @brief Serialized artifact or generated IR bytes.
         */
        clangrtcString output;

        /**
         * @brief Diagnostics and analysis output.
         */
        clangrtcString log;

        /**
         * @brief This operation's resolved entry expressions in request order.
         *
         * Compilation results do not include the input artifact's expression slots.
         * Inspect those independently through clangrtcGetArtifactInfo().
         */
        const clangrtcSymbol *symbols;

        /**
         * @brief Number of entries in symbols.
         */
        size_t symbolCount;

        /**
         * @brief Available stage measurements.
         */
        const clangrtcTiming *timings;

        /**
         * @brief Number of entries in timings.
         */
        size_t timingCount;

        /**
         * @brief Total operation duration in nanoseconds.
         *
         * Includes stage overhead and may exceed the sum of the reported stage
         * durations.
         */
        uint64_t totalElapsedNs;

        /** @brief Frontend setup, restoration, parsing, and IR-generation nanoseconds, including failed work. */
        uint64_t frontendNs;
    } clangrtcResultView;

    /**
     * @brief Compiler identity and supported capabilities.
     *
     * All referenced storage has static lifetime within the loaded library.
     * Capability strings describe supported features; they
     * do not guarantee that a particular source or target will compile.
     */
    typedef struct
    {
        /**
         * @brief clangrtc library version.
         */
        clangrtcString libraryVersion;

        /**
         * @brief Linked Clang version.
         */
        clangrtcString clangVersion;

        /**
         * @brief Linked LLVM version.
         */
        clangrtcString llvmVersion;

        /**
         * @brief Supported capability names.
         */
        const clangrtcString *capabilities;

        /**
         * @brief Number of capability entries.
         */
        size_t capabilityCount;
    } clangrtcCompilerInfo;

    /**
     * @brief Metadata of a loaded artifact.
     *
     * All referenced storage belongs to the artifact and remains valid until
     * clangrtcDestroyArtifact(). Inspection does not compile or rebuild it.
     */
    typedef struct
    {
        /**
         * @brief "pch", "bmi", "reduced-bmi", "cuda", or "preprocessed-cuda".
         */
        clangrtcString format;

        /**
         * @brief Preparing compiler identity.
         */
        clangrtcString compilerFingerprint;

        /**
         * @brief GPU target used for preparation.
         */
        clangrtcString target;

        /**
         * @brief Module name; empty otherwise.
         */
        clangrtcString moduleName;

        /**
         * @brief Total serialized package size in bytes.
         */
        size_t packageSize;

        /**
         * @brief Precomputed canonical XXH3-128 identity of the immutable artifact contents.
         */
        clangrtcString contentHash;

        /**
         * @brief Packaged dependency paths.
         */
        const clangrtcString *dependencies;

        /**
         * @brief Number of entries in dependencies.
         */
        size_t dependencyCount;

        /**
         * @brief Options used during preparation.
         */
        const clangrtcString *options;

        /**
         * @brief Number of entries in options.
         */
        size_t optionCount;

        /**
         * @brief Preparation expressions and resolved symbols in artifact slot order.
         *
         * These slots are fixed at preparation and independent of every compilation
         * request's slots. Their storage belongs to this artifact.
         */
        const clangrtcSymbol *symbols;

        /**
         * @brief Number of preparation expression slots in symbols.
         */
        size_t symbolCount;
    } clangrtcArtifactInfo;

    /**
     * @brief Initialize the compiler runtime.
     *
     * Optional: subsequent operations initialize the runtime automatically when needed.
     * Concurrent calls are safe; successful initialization is reused and failures may be retried.
     * @param callback Optional synchronous callback for initialization errors.
     * @param userData Caller data passed unchanged to callback.
     * @return CLANGRTC_SUCCESS, CLANGRTC_TOOLCHAIN_ERROR if required dependencies cannot be initialized,
     * CLANGRTC_OUT_OF_MEMORY, or CLANGRTC_INTERNAL_ERROR.
     */
    CLANGRTC_API clangrtcStatus clangrtcInit(clangrtcDiagnosticCallback callback, void *userData);

    /**
     * @brief Retrieves compiler identity and capabilities from static library storage.
     *
     * This query requires no compiler handle or compilation session. Its returned
     * strings and arrays remain valid for the lifetime of the loaded library.
     *
     * @param[out] info Receives metadata borrowed from static storage.
     * @return CLANGRTC_SUCCESS, or CLANGRTC_INVALID_ARGUMENT for a NULL argument.
     */
    CLANGRTC_API clangrtcStatus clangrtcGetInfo(clangrtcCompilerInfo *info);

    /**
     * @brief Queries a compiler/backend version without creating a compilation.
     *
     * @param component One of "clang", "llvm", or "nvvm" (explicit length).
     * @param version Borrowed version string with static lifetime, cleared on failure.
     * NVVM reports its API major/minor version, not the CUDA package version.
     * @return CLANGRTC_SUCCESS, CLANGRTC_INVALID_ARGUMENT for unknown components/NULL output,
     * CLANGRTC_OUT_OF_MEMORY on allocation failure,
     * CLANGRTC_TOOLCHAIN_ERROR for backend loading failure, or CLANGRTC_INTERNAL_ERROR for query failure.
     */
    CLANGRTC_API clangrtcStatus clangrtcGetVersion(clangrtcString component, clangrtcString *version);

    /**
     * @brief Prepares a self-contained artifact for subsequent in-memory use.
     *
     * Select the format with --artifact=pch, --artifact=bmi,
     * --artifact=reduced-bmi (the default), --artifact=cuda, or
     * --artifact=preprocessed-cuda. BMI preparation requires a named module in
     * the source. The result's output is the complete serialized package,
     * including dependencies, compiler identity, and semantic options.
     *
     * @param request Source, virtual headers, and optional entry expressions. Its
     * artifact must be NULL. Resolved expression slots are returned in the result
     * and stored in the artifact for inspection after loading.
     * @param options Complete options for this call; may be NULL when optionCount is zero.
     * @param optionCount Number of entries in options.
     * @param callback Optional synchronous diagnostic callback for this call. It
     * may be NULL; diagnostics remain available in the result log.
     * @param userData Caller-owned context passed unchanged to callback and borrowed
     * until this call returns. Callbacks must not throw across the C API boundary.
     * @param[out] result Receives an owned result. Source/option failures include
     * a diagnostic log. May be NULL on invalid arguments or allocation failure.
     * Destroy every non-NULL result even when the operation fails.
     * @return Operation status, also recorded in any returned result. Configuration failures use
     * CLANGRTC_TOOLCHAIN_ERROR; invalid entry slots use CLANGRTC_INVALID_EXPRESSION;
     * ordinary source errors use CLANGRTC_COMPILATION_FAILED.
     */
    CLANGRTC_API clangrtcStatus clangrtcPrepare(const clangrtcRequest *request, const clangrtcString *options,
                                                size_t optionCount, clangrtcDiagnosticCallback callback, void *userData,
                                                clangrtcResult *result);

    /**
     * @brief Copies and validates a serialized artifact from memory.
     *
     * Validates the package checksum and compiler compatibility. Loading neither
     * extracts files nor rebuilds compiler state. Packages must come from a
     * trusted producer: checksum validation is not authentication, and serialized
     * Clang state is not an untrusted-input sandbox.
     *
     * @param package Complete package bytes; the caller may release them on return.
     * @param callback Optional synchronous callback receiving loading diagnostics.
     * May be NULL; this operation creates no result log.
     * @param userData Caller-owned callback context borrowed until this call returns.
     * Callbacks must not throw across the C API boundary.
     * @param[out] artifact Receives an owned handle on success, or NULL on failure
     * when artifact is a valid output pointer.
     * @return CLANGRTC_SUCCESS, CLANGRTC_INVALID_ARGUMENT for a NULL required
     * argument, CLANGRTC_INVALID_ARTIFACT for malformed package data, CLANGRTC_INCOMPATIBLE_ARTIFACT for a compiler
     * mismatch, CLANGRTC_OUT_OF_MEMORY, or CLANGRTC_INTERNAL_ERROR.
     */
    CLANGRTC_API clangrtcStatus clangrtcLoadArtifact(clangrtcString package, clangrtcDiagnosticCallback callback,
                                                     void *userData, clangrtcArtifact *artifact);

    /**
     * @brief Retrieves metadata without constructing a compiler frontend.
     *
     * @param artifact Artifact to inspect.
     * @param[out] info Receives metadata borrowed from artifact.
     * @return CLANGRTC_SUCCESS, or CLANGRTC_INVALID_ARGUMENT for a NULL argument.
     */
    CLANGRTC_API clangrtcStatus clangrtcGetArtifactInfo(clangrtcArtifact artifact, clangrtcArtifactInfo *info);

    /**
     * @brief Releases an artifact after all operations borrowing it have returned.
     *
     * @param artifact Handle to release; NULL is allowed. All operations borrowing
     * this artifact must have returned before destruction.
     */
    CLANGRTC_API void clangrtcDestroyArtifact(clangrtcArtifact artifact);

    /**
     * @brief Compiles CUDA device code or native host C/C++, returning owned output bytes.
     *
     * Device output defaults to --emit=nvvm-ir-bitcode; --emit=nvvm-ir-text selects
     * textual IR. Use --lto-ir or --ptx-text for backend output through libNVVM.
     * Prepared PCH/BMI state can instantiate templates with new runtime-selected arguments.
     * Semantic options must match the prepared state; incompatible state is
     * rejected rather than rebuilt. When optionCount is zero, an artifact's
     * preparation options are inherited. Otherwise supply the complete option
     * array for this call; options are not appended to the artifact's settings.
     * Specify at most one output selector (--emit=..., --lto-ir, or --ptx-text).
     * Multiple selectors, including identical repetitions, return CLANGRTC_INVALID_OPTION.
     *
     * The LLVM IR dialect and bitcode encoding follow the linked LLVM release,
     * reported by clangrtcGetInfo(). Normalization supplies NVVM IR 2.0 and
     * debug metadata 3.2 when version metadata is absent; existing metadata is
     * preserved. These versions are distinct from the LLVM release and bitcode
     * format. Selecting text output does not downgrade the IR to an older dialect.
     * The downstream NVIDIA compiler must accept both the representation and its
     * IR features; successful LLVM verification here does not establish that.
     *
     * Pass multiple stage arguments using repeated --clang-option=ARG or --nvvm-option=ARG
     * options, or their two-element forms (--clang-option, ARG). Each occurrence passes
     * exactly one argument without shell splitting. Frontend flags participate in
     * artifact compatibility; NVVM backend flags do not. Pipeline, target, input/output,
     * and plugin overrides are reserved. NVVM options apply to LTO/PTX compilation;
     * host mode rejects NVVM options. nvJitLink options remain caller-owned.
     *
     * Host mode uses --host-only (C++20), --source=c (C17), or --source=c++
     * (C++20), with -std=... overrides. It defaults to llvm-ir-bitcode output;
     * --emit=llvm-ir-text and --emit=object are also supported. No CUDA path
     * is required. Host objects are position-independent relocatable native code;
     * the caller supplies linking and execution. Host C/C++ supports PCH preparation
     * and reuse; host C++ also supports BMI and reduced-BMI packages. Host/device
     * and C/C++ artifact settings must match. GPU targets and mixed host/CUDA
     * source are not supported in host mode.
     *
     * GPU execution is validated with Clang/LLVM 23.1.5 and libNVVM/nvJitLink
     * 13.5.37 on sm_86. The downstream flags are -nvvm-latest -gen-lto for
     * libNVVM and -nvvm-latest -lto for nvJitLink. Other combinations require
     * validation; these versions do not declare a minimum supported NVIDIA release.
     *
     * @param request Source, optional artifact, virtual headers, and entry
     * expressions. A PCH/BMI artifact requires consumer source. Source artifacts
     * use their packaged source when request.source.data is empty; otherwise,
     * their original source prefix precedes the supplied consumer source.
     * @param options Complete options for this call; may be NULL when optionCount is zero.
     * @param optionCount Number of entries in options.
     * @param callback Optional synchronous diagnostic callback for this call. It
     * may be NULL; diagnostics remain available in the result log.
     * @param userData Caller-owned context passed unchanged to callback and borrowed
     * until this call returns. Callbacks must not throw across the C API boundary.
     * @param[out] result Receives an owned result. Source/option failures include
     * a diagnostic log. May be NULL on invalid arguments or allocation failure.
     * Destroy every non-NULL result even when the operation fails.
     * @return Operation status, also recorded in any returned result. Configuration failures use
     * CLANGRTC_TOOLCHAIN_ERROR; invalid entry slots use CLANGRTC_INVALID_EXPRESSION;
     * ordinary source errors use CLANGRTC_COMPILATION_FAILED. Malformed prepared
     * bitcode returns CLANGRTC_INVALID_ARTIFACT; a failed link returns
     * CLANGRTC_LINK_FAILED.
     */
    CLANGRTC_API clangrtcStatus clangrtcCompile(const clangrtcRequest *request, const clangrtcString *options,
                                                size_t optionCount, clangrtcDiagnosticCallback callback, void *userData,
                                                clangrtcResult *result);

    /**
     * @brief Retrieves borrowed views of a completed operation result.
     *
     * The return value describes retrieval, while view.status describes the
     * original operation. Repeated calls preserve previously returned views.
     *
     * @param result Result to inspect, including one from a failed operation.
     * @param[out] view Receives views valid until result is destroyed.
     * @return CLANGRTC_SUCCESS, CLANGRTC_INVALID_ARGUMENT for a NULL argument, or
     * CLANGRTC_OUT_OF_MEMORY if view storage cannot be allocated.
     */
    CLANGRTC_API clangrtcStatus clangrtcGetResult(clangrtcResult result, clangrtcResultView *view);

    /**
     * @brief Releases a result and invalidates all views borrowed from it.
     *
     * @param result Handle to release; NULL is allowed.
     */
    CLANGRTC_API void clangrtcDestroyResult(clangrtcResult result);

    /**
     * @brief Returns a human-readable status description with static lifetime.
     *
     * @param status Status to describe; unknown values produce "unknown status".
     * @return Borrowed text view. No terminating NUL byte is required or counted.
     */
    CLANGRTC_API clangrtcString clangrtcStatusString(clangrtcStatus status);

#ifdef __cplusplus
} // extern "C"

#endif

#endif // CLANGRTC_H
