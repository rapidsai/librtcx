#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compile the trait/concept suites with NVRTC; optionally execute the runtime test."""

import argparse
import ctypes as ct
import ctypes.util as ctypes_util
import fnmatch
from pathlib import Path
import sys


TESTS = Path(__file__).resolve().parent
LIBRARY = TESTS.parent


def bind(library, name, arguments):
    function = getattr(library, name)
    function.argtypes = arguments
    function.restype = ct.c_int
    return function


class Nvrtc:
    def __init__(self, path):
        self.library = ct.CDLL(path)
        self.version = bind(self.library, "nvrtcVersion", [ct.POINTER(ct.c_int)] * 2)
        self.create = bind(self.library, "nvrtcCreateProgram", [
            ct.POINTER(ct.c_void_p), ct.c_char_p, ct.c_char_p, ct.c_int,
            ct.c_void_p, ct.c_void_p,
        ])
        self.compile_program = bind(self.library, "nvrtcCompileProgram", [
            ct.c_void_p, ct.c_int, ct.POINTER(ct.c_char_p),
        ])
        self.destroy = bind(self.library, "nvrtcDestroyProgram", [ct.POINTER(ct.c_void_p)])
        self.error_string = self.library.nvrtcGetErrorString
        self.error_string.argtypes = [ct.c_int]
        self.error_string.restype = ct.c_char_p
        for kind in ("ProgramLog", "PTX"):
            bind(self.library, f"nvrtcGet{kind}Size", [ct.c_void_p, ct.POINTER(ct.c_size_t)])
            bind(self.library, f"nvrtcGet{kind}", [ct.c_void_p, ct.c_void_p])
        major, minor = ct.c_int(), ct.c_int()
        self.check(self.version(ct.byref(major), ct.byref(minor)))
        self.label = f"{major.value}.{minor.value} ({path})"

    def check(self, status):
        if status:
            raise RuntimeError(self.error_string(status).decode())

    def output(self, program, kind):
        size = ct.c_size_t()
        self.check(getattr(self.library, f"nvrtcGet{kind}Size")(program, ct.byref(size)))
        buffer = ct.create_string_buffer(size.value)
        self.check(getattr(self.library, f"nvrtcGet{kind}")(program, buffer))
        return buffer.value

    def compile(self, source, name, options):
        program = ct.c_void_p()
        self.check(self.create(ct.byref(program), source.encode(), name.encode(), 0, None, None))
        try:
            arguments = (ct.c_char_p * len(options))(*(option.encode() for option in options))
            status = self.compile_program(program, len(options), arguments)
            log = self.output(program, "ProgramLog").decode()
            if status:
                raise RuntimeError(f"{self.error_string(status).decode()}\n{log}")
            return self.output(program, "PTX")
        finally:
            self.check(self.destroy(ct.byref(program)))


class Device:
    """Minimal Driver API wrapper for the runtime_checks(int*) test protocol."""

    def __init__(self, ordinal):
        self.driver = ct.CDLL(ctypes_library("cuda"))
        self.error_string = self.driver.cuGetErrorString
        self.error_string.argtypes = [ct.c_int, ct.POINTER(ct.c_char_p)]
        self.error_string.restype = ct.c_int
        signatures = {
            "cuInit": [ct.c_uint],
            "cuDeviceGet": [ct.POINTER(ct.c_int), ct.c_int],
            "cuDevicePrimaryCtxRetain": [ct.POINTER(ct.c_void_p), ct.c_int],
            "cuDevicePrimaryCtxRelease_v2": [ct.c_int],
            "cuCtxSetCurrent": [ct.c_void_p],
            "cuModuleLoadData": [ct.POINTER(ct.c_void_p), ct.c_void_p],
            "cuModuleUnload": [ct.c_void_p],
            "cuModuleGetFunction": [ct.POINTER(ct.c_void_p), ct.c_void_p, ct.c_char_p],
            "cuMemAlloc_v2": [ct.POINTER(ct.c_uint64), ct.c_size_t],
            "cuMemFree_v2": [ct.c_uint64],
            "cuMemsetD32_v2": [ct.c_uint64, ct.c_uint, ct.c_size_t],
            "cuMemcpyDtoH_v2": [ct.c_void_p, ct.c_uint64, ct.c_size_t],
            "cuCtxSynchronize": [],
            "cuLaunchKernel": [ct.c_void_p] + [ct.c_uint] * 7 + [ct.c_void_p] * 3,
        }
        for name, signature in signatures.items():
            setattr(self, name, bind(self.driver, name, signature))
        self.check(self.cuInit(0))
        self.device = ct.c_int()
        self.check(self.cuDeviceGet(ct.byref(self.device), ordinal))
        self.context = ct.c_void_p()
        self.check(self.cuDevicePrimaryCtxRetain(ct.byref(self.context), self.device))
        try:
            self.check(self.cuCtxSetCurrent(self.context))
        except Exception:
            self.close()
            raise

    def check(self, status):
        if status:
            message = ct.c_char_p()
            self.error_string(status, ct.byref(message))
            detail = message.value.decode() if message.value else str(status)
            raise RuntimeError(f"CUDA Driver: {detail}")

    def close(self):
        self.check(self.cuDevicePrimaryCtxRelease_v2(self.device))

    def run(self, ptx):
        module, function = ct.c_void_p(), ct.c_void_p()
        allocation = ct.c_uint64()
        image = ct.create_string_buffer(ptx)
        self.check(self.cuModuleLoadData(ct.byref(module), image))
        try:
            self.check(self.cuModuleGetFunction(ct.byref(function), module, b"runtime_checks"))
            self.check(self.cuMemAlloc_v2(ct.byref(allocation), 4))
            try:
                self.check(self.cuMemsetD32_v2(allocation, 0xFFFFFFFF, 1))
                arguments = (ct.c_void_p * 1)(ct.addressof(allocation))
                self.check(self.cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0,
                                               None, arguments, None))
                self.check(self.cuCtxSynchronize())
                failures = ct.c_int()
                self.check(self.cuMemcpyDtoH_v2(ct.byref(failures), allocation, 4))
                if failures.value:
                    raise RuntimeError(f"runtime_checks reported {failures.value} failures")
            finally:
                self.check(self.cuMemFree_v2(allocation))
        finally:
            self.check(self.cuModuleUnload(module))


def ctypes_library(name):
    path = ctypes_util.find_library(name)
    if not path:
        raise RuntimeError(f"Cannot locate {name}; pass --nvrtc or configure the library search path")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nvrtc", action="append", help="NVRTC shared library; repeat for multiple versions")
    parser.add_argument("--reference-includes", type=Path,
                        help="CCCL include root; also compile the shared tests against libcudacxx")
    parser.add_argument("--filter", default="*", help="Test filename glob")
    parser.add_argument("--architecture", default="compute_80")
    parser.add_argument("--run", action="store_true", help="Execute runtime.cu on a CUDA GPU")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    tests = sorted(TESTS.glob("*.cu"))
    tests = [test for test in tests if fnmatch.fnmatch(test.name, args.filter)]
    if not tests:
        parser.error("No tests match --filter")
    if args.run and not any(test.name == "runtime.cu" for test in tests):
        parser.error("--run requires runtime.cu to match --filter")
    if args.reference_includes and not (args.reference_includes / "cuda/std/concepts").is_file():
        parser.error("--reference-includes must contain cuda/std/concepts")
    device = Device(args.device) if args.run else None
    passed = failed = 0
    try:
        for path in args.nvrtc or [ctypes_library("nvrtc")]:
            compiler = Nvrtc(path)
            print(f"NVRTC {compiler.label}", flush=True)
            modes = ["native", "aliases"]
            if args.reference_includes:
                modes.append("libcudacxx")
            for test in tests:
                for mode in modes:
                    options = ["-std=c++20", "--device-int128",
                               "--device-as-default-execution-space",
                               f"--gpu-architecture={args.architecture}", f"-I{TESTS}"]
                    if mode == "libcudacxx":
                        options += ["-DTEST_LIBCUDACXX", f"-I{args.reference_includes.resolve()}"]
                    else:
                        options += ["--minimal", f"-I{LIBRARY}", f"-I{LIBRARY / 'nvcxx'}"]
                    if mode == "aliases":
                        options.append("-DNVCXX_AS_STD")
                    label = f"{test.name} [{mode}]"
                    try:
                        ptx = compiler.compile(test.read_text(), test.name, options)
                        if device and test.name == "runtime.cu":
                            device.run(ptx)
                        passed += 1
                        print(f"  PASS {label}", flush=True)
                    except RuntimeError as error:
                        failed += 1
                        print(f"  FAIL {label}\n{error}", flush=True)
    finally:
        if device:
            device.close()
    print(f"{passed} passed, {failed} failed; GPU execution {'enabled' if args.run else 'disabled'}")
    return bool(failed)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
