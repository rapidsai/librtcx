#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORMAT_CONFIG="${SCRIPT_DIR}/cmake_config_format.json"
LINT_CONFIG="${SCRIPT_DIR}/cmake_config_lint.json"
USAGE="bash run-cmake-format.sh {cmake-format,cmake-lint} infile [infile ...]"

if [[ $# -lt 2 ]]; then
  echo "Usage: ${USAGE}"
  exit 1
fi

if [[ $1 == "cmake-format" ]]; then
  for cmake_file in "${@:2}"; do
    cmake-format --in-place --config-files "${FORMAT_CONFIG}" -- "${cmake_file}" || exit $?
  done
elif [[ $1 == "cmake-lint" ]]; then
  OUTPUT=$(cmake-lint --config-files "${FORMAT_CONFIG}" "${LINT_CONFIG}" -- "${@:2}")
  status=$?

  if ! [ ${status} -eq 0 ]; then
    echo "${OUTPUT}"
  fi

  exit ${status}
else
  echo "Usage: ${USAGE}"
  exit 1
fi
