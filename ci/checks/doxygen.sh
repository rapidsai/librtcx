#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "usage: $0 [output-directory]"
  exit 2
fi

doxygen_executable=${DOXYGEN_EXECUTABLE:-doxygen}
if ! command -v "${doxygen_executable}" >/dev/null 2>&1; then
  echo "warning: doxygen is not installed; skipping documentation verification"
  exit 0
fi

project_root=$(realpath "$(dirname "${BASH_SOURCE[0]}")/../..")
docs_dir="${project_root}/docs"
remove_docs_output=false
if [[ $# -eq 0 ]]; then
  docs_output=$(mktemp -d)
  remove_docs_output=true
else
  docs_output=$1
fi
stderr_log="${docs_output}/doxygen.stderr"

cleanup() {
  if [[ "${remove_docs_output}" == true ]]; then
    cmake -E remove_directory "${docs_output}"
  fi
}
trap cleanup EXIT

cmake -E make_directory "${docs_output}"
rtcx_version=$(tr -d '[:space:]' < "${project_root}/VERSION")

set +e
(
  cd "${docs_dir}"
  RTCX_VERSION="${rtcx_version}" RTCX_DOCS_OUTPUT="${docs_output}" \
    "${doxygen_executable}" Doxyfile >/dev/null 2>"${stderr_log}"
)
doxygen_status=$?
set -e

# Doxygen 1.9.1 reports incomplete comments through WARN_IF_DOC_ERROR even when
# WARN_IF_UNDOCUMENTED is disabled. Ignore only missing-documentation warnings;
# malformed commands and references remain fatal.
filtered_stderr="${docs_output}/doxygen.filtered.stderr"
sed -E -e '/warning: (Compound|Member) .* is not documented[.]$/d' \
  -e '/warning: parameters of member .* are not \(all\) documented$/d' \
  -e '/warning: return type of member .* is not documented$/d' "${stderr_log}" >"${filtered_stderr}"
mv "${filtered_stderr}" "${stderr_log}"

if [[ ${doxygen_status} -ne 0 ]]; then
  cat "${stderr_log}"
  echo "error: doxygen exited with status ${doxygen_status}"
  exit "${doxygen_status}"
fi

if [[ -s "${stderr_log}" ]]; then
  cat "${stderr_log}"
  echo "error: doxygen emitted warnings"
  exit 1
fi

if [[ ! -f "${docs_output}/html/index.html" ]]; then
  echo "error: doxygen did not generate html/index.html"
  exit 1
fi
