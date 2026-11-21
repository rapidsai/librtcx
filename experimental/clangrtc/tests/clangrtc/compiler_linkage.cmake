# SPDX-License-Identifier: Apache-2.0

# @file tests/clangrtc/compiler_linkage.cmake
#
# @brief Check configuration-time acceptance and rejection of compiler dependency graphs.
#
# Runs the synthetic package fixture for CASE and requires either static-linkage acceptance or the
# expected archive-policy diagnostic.

execute_process(COMMAND "${CMAKE_COMMAND}" -S "${SOURCE}" -B "${WORK}" "-DCASE=${CASE}"
                        "-DMODULE=${MODULE}" RESULT_VARIABLE result OUTPUT_VARIABLE output
                ERROR_VARIABLE error)
if(CASE STREQUAL "static")
  if(NOT result EQUAL 0)
    message(FATAL_ERROR "Static compiler dependencies were rejected:\n${output}\n${error}")
  endif()
else()
  if(result EQUAL 0)
    message(FATAL_ERROR "Invalid compiler linkage ${CASE} was accepted")
  endif()
  # CMake wraps diagnostics according to the length of the configured paths.
  string(REGEX REPLACE "[ \t\r\n]+" " " diagnostic "${error}")
  if(NOT
     diagnostic
     MATCHES
     "requires static LLVM/Clang|must reference a static archive|must be an existing static archive|is not installed"
  )
    message(FATAL_ERROR "Unexpected failure for ${CASE}:\n${output}\n${error}")
  endif()
endif()
