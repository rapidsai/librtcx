/* SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iostream>
#include <llvm/AsmParser/Parser.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>
#include <nvvm.h>
#include <string>

int main()
{
    llvm::LLVMContext context;
    llvm::SMDiagnostic diagnostic;
    auto module = llvm::parseAssemblyString(R"(
    target datalayout = "e-p:64:64:64-p6:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-f128:128:128-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64-a:8:8"
    target triple = "nvptx64-nvidia-cuda"
    define ptx_kernel void @probe(ptr addrspace(1) %out) {
      store i32 42, ptr addrspace(1) %out
      ret void
    }
    !nvvmir.version = !{!0}
    !0 = !{i32 2, i32 0, i32 3, i32 2}
  )",
                                            diagnostic, context);

    if (!module)
    {
        return 1;
    }

    std::string bytes;
    llvm::raw_string_ostream stream(bytes);
    llvm::WriteBitcodeToFile(*module, stream);

    for (int count : {3, 2})
    {
        nvvmProgram program{};

        if (nvvmCreateProgram(&program) != NVVM_SUCCESS)
        {
            return 2;
        }

        auto status = nvvmAddModuleToProgram(program, bytes.data(), bytes.size(), "compatibility");
        char const *options[]{"-arch=compute_86", "-nvvm-latest", "-gen-lto"};

        if (status == NVVM_SUCCESS)
        {
            status = nvvmCompileProgram(program, count, options);
        }

        size_t size{};
        nvvmGetProgramLogSize(program, &size);
        std::string log(size, '\0');

        if (size)
        {
            nvvmGetProgramLog(program, log.data());
        }

        nvvmDestroyProgram(&program);

        if (status != NVVM_SUCCESS)
        {
            std::cerr << nvvmGetErrorString(status) << '\n' << log;
            return 3;
        }
    }

    return 0;
}
