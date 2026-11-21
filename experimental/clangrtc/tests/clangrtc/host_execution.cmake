# SPDX-License-Identifier: Apache-2.0

# @file tests/clangrtc/host_execution.cmake
#
# @brief Compile host objects and execute native programs linked from their output.
#
# Checks C/C++ code and consumers of PCH, BMI, and reduced-BMI artifacts after deleting the prepared
# source files.

file(MAKE_DIRECTORY "${WORK}")
file(WRITE "${WORK}/answer.cpp"
     [=[
#include <vector>

extern "C" int cpp_answer(int x)
{
    std::vector<int> values{x, 2};
    return values[0] + values[1];
}
]=])
file(WRITE "${WORK}/answer.c"
     [=[
#include <stdint.h>

int c_answer(int x)
{
    return (int32_t)x + 1;
}
]=])
file(WRITE "${WORK}/main.cpp"
     [=[
extern "C" int cpp_answer(int);
extern "C" int c_answer(int);

int main()
{
    return cpp_answer(40) == 42 && c_answer(41) == 42 ? 0 : 1;
}
]=])

foreach(language IN ITEMS c c++)
  if(language STREQUAL "c")
    set(source answer.c)
    set(object answer_c.o)
  else()
    set(source answer.cpp)
    set(object answer_cpp.o)
  endif()
  execute_process(COMMAND "${CLI}" compile "${WORK}/${source}" -o "${WORK}/${object}" --
                          "--source=${language}" --emit=object --cuda-path=/missing/cuda
                  RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE errors)
  if(NOT status STREQUAL "0")
    message(FATAL_ERROR "Host ${language} compilation failed: ${status}\n${output}\n${errors}")
  endif()
endforeach()
execute_process(COMMAND "${CXX}" "${WORK}/main.cpp" "${WORK}/answer_cpp.o" "${WORK}/answer_c.o" -o
                        "${WORK}/host_test" RESULT_VARIABLE status OUTPUT_VARIABLE output
                ERROR_VARIABLE errors)
if(NOT status STREQUAL "0")
  message(FATAL_ERROR "Host linking failed: ${status}\n${output}\n${errors}")
endif()
execute_process(COMMAND "${WORK}/host_test" RESULT_VARIABLE status)
if(NOT status STREQUAL "0")
  message(FATAL_ERROR "Host executable returned ${status}")
endif()

# Each host artifact must produce an object usable by the ordinary native linker.
foreach(kind IN ITEMS pch bmi reduced-bmi)
  if(kind STREQUAL "pch")
    set(prefix "#include <array>\n")
    set(import "")
  else()
    set(prefix "module;\n#include <array>\nexport module host_execution;\nexport ")
    set(import "import host_execution;\n")
  endif()
  string(APPEND prefix "template<class T> T scale(T x) { std::array<T,1> v{x}; return v[0]*3; }\n")
  file(WRITE "${WORK}/prefix.cpp" "${prefix}")
  execute_process(COMMAND "${CLI}" prepare "${WORK}/prefix.cpp" -o "${WORK}/${kind}.pkg" --
                          --host-only "--artifact=${kind}" RESULT_VARIABLE status
                  OUTPUT_VARIABLE output ERROR_VARIABLE errors)
  if(NOT status STREQUAL "0")
    message(FATAL_ERROR "Host ${kind} preparation failed: ${status}\n${output}\n${errors}")
  endif()
  file(REMOVE "${WORK}/prefix.cpp")
  file(WRITE "${WORK}/consumer.cpp"
       "${import}extern \"C\" int from_artifact(int x) {return scale(x);}\n")
  execute_process(COMMAND "${CLI}" compile "${WORK}/consumer.cpp" -o "${WORK}/consumer.o"
                          --artifact-file "${WORK}/${kind}.pkg" -- --host-only --emit=object
                  RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE errors)
  if(NOT status STREQUAL "0")
    message(FATAL_ERROR "Host ${kind} consumption failed: ${status}\n${output}\n${errors}")
  endif()
  file(WRITE "${WORK}/caller.cpp"
       "extern \"C\" int from_artifact(int); int main() {return from_artifact(14)==42 ? 0 : 1;}\n")
  execute_process(COMMAND "${CXX}" "${WORK}/caller.cpp" "${WORK}/consumer.o" -o
                          "${WORK}/artifact_test" RESULT_VARIABLE status OUTPUT_VARIABLE output
                  ERROR_VARIABLE errors)
  if(NOT status STREQUAL "0")
    message(FATAL_ERROR "Host ${kind} linking failed: ${status}\n${output}\n${errors}")
  endif()
  execute_process(COMMAND "${WORK}/artifact_test" RESULT_VARIABLE status)
  if(NOT status STREQUAL "0")
    message(FATAL_ERROR "Host ${kind} executable returned ${status}")
  endif()
endforeach()
