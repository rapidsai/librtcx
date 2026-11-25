/* SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef CURTC_H
#define CURTC_H

/**
 * @file curtc.h
 * @brief Compile, link, cache, and load a reference-counted CUDA library.
 *
 * Each cache owns its configuration and supports concurrent compilation. Inputs
 * are borrowed during synchronous calls. Successful compilation returns one
 * library reference owning its CUDA library, independently of cache lifetime and
 * eviction. Keep a reference alive until all GPU work using its kernels completes.
 * The caller owns launches, streams, and synchronization.
 *
 * All text/byte ranges use clangrtcString with explicit sizes and no required
 * NUL terminator. Callers must supply valid pointer/count ranges for arrays and
 * text/byte views; these ranges are assumed valid. Zero-initialize requests. Output handles are cleared before
 * validation; pass an empty slot. No C++ exception crosses this interface.
 */
#include <clangrtc.h>
#include <stdbool.h>

/**
 * @brief Borrowed CUDA kernel handle; compatible with the declaration in cuda.h.
 */
typedef struct CUkern_st *CUkernel;

#ifdef __cplusplus
extern "C"
{
#endif

    /**
     * @brief Owned compilation/loading cache with an independent configuration.
     */
    typedef struct curtcCacheImpl *curtcCache;

    /**
     * @brief Reference-counted loaded library containing the kernels selected by a request.
     */
    typedef struct curtcLibraryImpl *curtcLibrary;

    /**
     * @brief Representation of a kernel or device-code fragment.
     */
    typedef enum
    {
        /**
         * @brief CUDA C++ source input.
         */
        CURTC_INPUT_TYPE_CUDA = 0,

        /**
         * @brief Ready, immutable clangrtc artifact in source.artifact.
         */
        CURTC_INPUT_TYPE_ARTIFACT = 1,

        /**
         * @brief NVIDIA LTO IR.
         */
        CURTC_INPUT_TYPE_LTO = 2,

        /**
         * @brief PTX assembly.
         */
        CURTC_INPUT_TYPE_PTX = 3,

        /**
         * @brief Finalized CUDA cubin.
         */
        CURTC_INPUT_TYPE_CUBIN = 4

    } curtcInputType;

    /**
     * @brief Kernel selector with an optional application identity.
     */
    typedef struct
    {
        /**
         * @brief CUDA expression for source/artifact inputs, or exact symbol for binary inputs.
         */
        clangrtcString name;

        /**
         * @brief User-specified identity copied into the result; does not affect code caching.
         */
        clangrtcString id;
    } curtcKernelSelection;

    /**
     * @brief Borrowed input; type selects source/artifact fields or precompiled data.
     */
    typedef struct
    {
        /**
         * @brief Actual representation of this input.
         */
        curtcInputType type;

        /**
         * @brief CUDA source/consumer, virtual headers, and a borrowed ready artifact for ARTIFACT inputs.
         *
         * Load PCH/BMI state once with clangrtcLoadArtifact and set source.artifact for
         * CURTC_INPUT_TYPE_ARTIFACT. Keep the handle alive until calls borrowing it return.
         * curtc uses its precomputed identity without validation, hashing, or loading. Expressions
         * retain supported definitions in addition to the kernels selected by entries.
         */
        clangrtcRequest source;

        /**
         * @brief LTO, PTX, or cubin bytes; ignored for CUDA and ARTIFACT inputs.
         */
        clangrtcString data;

        /**
         * @brief Ordered clangrtc options for source inputs; frontend options are inherited.
         */
        const clangrtcString *options;

        /**
         * @brief Number of options; options are ignored for precompiled inputs.
         */
        size_t optionCount;

        /**
         * @brief Ordered kernel expressions or binary symbols selected from this input.
         *
         * Each selection retains its application identity and source metadata in the
         * returned library. NULL with entryCount zero contributes code without adding
         * metadata rows or instantiating kernels through this array. Exported kernels
         * remain retrievable by symbol through curtcGetKernel().
         */
        const curtcKernelSelection *entries;

        /**
         * @brief Number of selected kernels; inputs without entries still contribute device code.
         */
        size_t entryCount;
    } curtcInput;

    /**
     * @brief Borrowed kernel metadata, valid for the library reference's lifetime.
     */
    typedef struct
    {
        /**
         * @brief User-specified identity, including for opaque AOT/LTO inputs; empty if none was supplied.
         */
        clangrtcString id;

        /**
         * @brief Original CUDA expression used to select and instantiate this kernel; empty for binary/link-only
         * inputs.
         */
        clangrtcString expression;

        /**
         * @brief Qualified CUDA source name with template arguments; empty for binary inputs.
         */
        clangrtcString cudaName;

        /**
         * @brief Exact resolved symbol used for CUDA kernel lookup.
         */
        clangrtcString symbolName;
    } curtcKernel;

    /**
     * @brief Ordered device-code inputs finalized together into one loaded library.
     *
     * Each input may select kernels independently. Enumeration follows input order,
     * then entry order within each input. Inputs without entries still contribute
     * code; no entries yields empty metadata without restricting symbol lookup.
     */
    typedef struct
    {
        /**
         * @brief Inputs supplying source, artifacts, LTO IR, PTX, or cubin code.
         */
        const curtcInput *inputs;

        /**
         * @brief Number of inputs; must be nonzero.
         */
        size_t inputCount;

        /**
         * @brief Explicit sm_NN compilation/link target; optional when loading a sole cubin directly.
         */
        clangrtcString target;

        /**
         * @brief Additional nvJitLink options; target and output selection are owned by the workflow.
         */
        const clangrtcString *linkOptions;

        /**
         * @brief Number of link options; NULL is allowed for a zero count.
         */
        size_t linkOptionCount;
    } curtcRequest;

    /**
     * @brief Cache policy; obtain defaults with curtcDefaultConfiguration().
     */
    typedef struct
    {
        /**
         * @brief Preferred cache root; absent or unusable roots fall back to XDG, home, and temporary caches.
         *
         * v1 is appended, without a process-specific subdirectory. Processes sharing a root reuse disk entries.
         * If no usable directory is found, disk caching is disabled.
         */
        clangrtcString cacheDirectory;

        /**
         * @brief Budget shared by ready code and loaded-kernel retention; excludes in-flight work and caller-held
         * libraries.
         */
        uint64_t memoryBytes;

        /**
         * @brief Budget for published disk entries; temporary writes may temporarily exceed it.
         */
        uint64_t diskBytes;

        /**
         * @brief Enable ready memory retention; in-flight work is shared regardless.
         */
        bool memoryCache;

        /**
         * @brief Enable disk reuse; disk failures fall back to compilation.
         */
        bool diskCache;

        /**
         * @brief Optional cache-warning callback, called outside locks; must not throw.
         */
        clangrtcDiagnosticCallback diagnostic;

        /**
         * @brief Callback context; must remain valid for the cache lifetime.
         */
        void *userData;
    } curtcConfiguration;

    /**
     * @brief Cumulative cache counters; concurrent work can advance fields during the snapshot.
     */
    typedef struct
    {
        /**
         * @brief Ready results reused from memory.
         */
        uint64_t memoryHits;

        /**
         * @brief Validated disk results reused.
         */
        uint64_t diskHits;

        /**
         * @brief Requests requiring computation after cache lookup.
         */
        uint64_t misses;

        /**
         * @brief Requests waiting for identical in-flight work.
         */
        uint64_t sharedRequests;

        /**
         * @brief Reported disk failures, including corrupt entries.
         */
        uint64_t diskErrors;

        /**
         * @brief Compilations of source/artifact inputs with selected kernels attempted.
         */
        uint64_t kernelCompilations;

        /**
         * @brief Compilations of source/artifact inputs without selected kernels; excludes dependency scans.
         */
        uint64_t fragmentCompilations;

        /**
         * @brief nvJitLink finalizations attempted; excludes direct cubin retrieval.
         */
        uint64_t links;
    } curtcCacheStatistics;

    /**
     * @brief Statistics attributed only to one curtcGet() call, including failed work.
     */
    typedef struct
    {
        /**
         * @brief Ready stage results reused from memory.
         */
        uint64_t memoryHits;

        /**
         * @brief Stage results reused from disk.
         */
        uint64_t diskHits;

        /**
         * @brief Stage results computed by this call.
         */
        uint64_t misses;

        /**
         * @brief Stages satisfied by waiting for another request.
         */
        uint64_t sharedRequests;

        /**
         * @brief Disk errors encountered by this call.
         */
        uint64_t diskErrors;

        /**
         * @brief Inputs compiled with selected kernels, independent of the number of entries per input.
         */
        uint64_t kernelCompilations;

        /**
         * @brief Inputs compiled without selected kernels; excludes preprocessing scans.
         */
        uint64_t fragmentCompilations;

        /**
         * @brief Final linking operations attempted.
         */
        uint64_t links;

        /**
         * @brief CUDA library-load attempts by this call.
         */
        uint64_t libraryLoads;

        /**
         * @brief Elapsed nanoseconds, including waits and source dependency scans.
         */
        uint64_t totalNs;

        /** @brief Sum of this call's frontend setup, parsing, and IR-generation nanoseconds; excludes linking. */
        uint64_t frontendNs;
    } curtcCompileStatistics;

    /**
     * @brief Initialize the CUDA compilation and linking runtime.
     *
     * Optional: requests initialize the runtime automatically when needed. Concurrent calls
     * are safe, successful initialization is reused, and failed initialization may be retried.
     * Does not create a CUDA context.
     * @param callback Optional synchronous callback for initialization errors.
     * @param userData Caller data passed unchanged to callback.
     * @return CLANGRTC_SUCCESS or a clangrtcStatus describing backend initialization failure.
     */
    CLANGRTC_API clangrtcStatus curtcInit(clangrtcDiagnosticCallback callback, void *userData);

    /**
     * @brief Get default cache policy without creating a cache or loading the CUDA driver.
     * @param configuration Destination for defaults.
     * @return CLANGRTC_SUCCESS, or CLANGRTC_INVALID_ARGUMENT for NULL output.
     */
    CLANGRTC_API clangrtcStatus curtcDefaultConfiguration(curtcConfiguration *configuration);

    /**
     * @brief Create an independent cache; directory selection and driver loading are lazy.
     * @param configuration Policy copied into the cache, or NULL for defaults.
     * @param output Destination for one owned cache, cleared before validation.
     * @return CLANGRTC_SUCCESS or an existing clangrtcStatus failure code.
     */
    CLANGRTC_API clangrtcStatus curtcCreateCache(const curtcConfiguration *configuration, curtcCache *output);

    /**
     * @brief Destroy a cache without invalidating caller-owned libraries; NULL is a no-op.
     * @param cache Cache to destroy after its concurrent API calls have returned.
     */
    CLANGRTC_API void curtcDestroyCache(curtcCache cache);

    /**
     * @brief Retrieve or build a CUDA library with its kernel handles populated before publication.
     * @param cache Non-NULL cache for compilation, linking, and library reuse.
     * @param request Ordered inputs with optional kernel selections on each input.
     * @param callback Optional synchronous failure diagnostic callback; must not throw.
     * @param userData Caller context borrowed during failure diagnostics.
     * @param output Destination for one library reference, cleared before validation.
     * @param statistics Optional per-call statistics, populated on success and failure.
     * @return CLANGRTC_SUCCESS or an existing clangrtcStatus failure code. Driver
     *         failures return CLANGRTC_TOOLCHAIN_ERROR with a cuda-stage diagnostic.
     */
    CLANGRTC_API clangrtcStatus curtcGet(curtcCache cache, const curtcRequest *request,
                                         clangrtcDiagnosticCallback callback, void *userData, curtcLibrary *output,
                                         curtcCompileStatistics *statistics);

    /**
     * @brief Enumerate selected kernel metadata in request order without driver calls.
     * @param library Live library reference owning the returned strings, independent of cache lifetime.
     * @param count Required output count; when kernels is non-NULL, its input value is the array capacity.
     *              Receives the required count even if capacity is insufficient; cleared for an invalid library.
     * @param kernels Caller-owned array, or NULL to query the required count. Insufficient capacity leaves the
     *                array unchanged. Strings need not be NUL-terminated and borrow the library.
     * @return CLANGRTC_SUCCESS, or CLANGRTC_INVALID_ARGUMENT for a NULL library/count or insufficient capacity.
     */
    CLANGRTC_API clangrtcStatus curtcEnumerateKernels(curtcLibrary library, size_t *count, curtcKernel *kernels);

    /**
     * @brief Retrieve a preloaded CUDA kernel by its exact library symbol without driver calls or locking.
     * @param library Live library reference that must outlive all uses, including asynchronous launches.
     * @param symbol Exact exported symbol, such as curtcKernel::symbolName; must be nonempty without NUL bytes.
     *               The symbol need not appear in the request's selected metadata.
     * @param output Destination for a borrowed CUkernel, cleared before validation.
     * @return CLANGRTC_SUCCESS, CLANGRTC_INVALID_ARGUMENT for invalid inputs, CLANGRTC_TOOLCHAIN_ERROR for
     *         an unavailable symbol, or CLANGRTC_OUT_OF_MEMORY.
     */
    CLANGRTC_API clangrtcStatus curtcGetKernel(curtcLibrary library, clangrtcString symbol, CUkernel *output);

    /**
     * @brief Add one library reference; NULL is a no-op.
     * @param library Live library reference to retain.
     */
    CLANGRTC_API void curtcRetainLibrary(curtcLibrary library);

    /**
     * @brief Release one library reference; NULL is a no-op.
     * @param library Reference to release after its GPU work completes.
     */
    CLANGRTC_API void curtcReleaseLibrary(curtcLibrary library);

    /**
     * @brief Snapshot counters for one cache; does not initialize disk storage or CUDA.
     * @param cache Non-NULL cache to inspect.
     * @param statistics Destination cleared before validation.
     * @return CLANGRTC_SUCCESS, or CLANGRTC_INVALID_ARGUMENT for NULL inputs.
     */
    CLANGRTC_API clangrtcStatus curtcGetStatistics(curtcCache cache, curtcCacheStatistics *statistics);

#ifdef __cplusplus
}

#endif
#endif
