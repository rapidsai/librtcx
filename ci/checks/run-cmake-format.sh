#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(realpath "${SCRIPT_DIR}"/../..)"
FORMAT_CONFIG="${SCRIPT_DIR}/cmake_config_format.json"
LINT_CONFIG="${SCRIPT_DIR}/cmake_config_lint.json"

# Look for the rapids-cmake shared format file that defines parse rules for
# rapids-cmake functions (rapids_find_package, rapids_cpm_find, etc.).
# In CI this is downloaded and set via RAPIDS_CMAKE_FORMAT_FILE.
# Locally it may exist if the user has fetched it.
if [ -z "${RAPIDS_CMAKE_FORMAT_FILE+x}" ]; then
  RAPIDS_CMAKE_FORMAT_FILE="${REPO_ROOT}/cmake-format-rapids-cmake.json"
fi

RAPIDS_CMAKE_FORMAT_ARGS=()
if [ -f "${RAPIDS_CMAKE_FORMAT_FILE}" ]; then
  RAPIDS_CMAKE_FORMAT_ARGS=("${RAPIDS_CMAKE_FORMAT_FILE}")
fi

USAGE="bash run-cmake-format.sh {cmake-format,cmake-lint} infile [infile ...]"

if [[ $# -lt 2 ]]; then
  echo "Usage: ${USAGE}"
  exit 1
fi

if [[ $1 == "cmake-format" ]]; then
  for cmake_file in "${@:2}"; do
    cmake-format --in-place --config-files "${RAPIDS_CMAKE_FORMAT_ARGS[@]}" "${FORMAT_CONFIG}" -- "${cmake_file}" || exit $?
  done
elif [[ $1 == "cmake-lint" ]]; then
  OUTPUT=$(cmake-lint --config-files "${RAPIDS_CMAKE_FORMAT_ARGS[@]}" "${FORMAT_CONFIG}" "${LINT_CONFIG}" -- "${@:2}")
  status=$?

  if ! [ ${status} -eq 0 ]; then
    echo "${OUTPUT}"
  fi

  exit ${status}
else
  echo "Usage: ${USAGE}"
  exit 1
fi
