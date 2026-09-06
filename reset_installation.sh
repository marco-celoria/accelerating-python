#!/bin/bash

set -euo pipefail

BASE_DIR="${1:-$PWD}"
BASE_DIR="$(readlink -f "${BASE_DIR}")"

echo "BASE_DIR=${BASE_DIR}"

VENV_DIR="${BASE_DIR}/pygpu_cuda122"
SOURCE_CACHE="${BASE_DIR}/sources"
WHEEL_DIR="${BASE_DIR}/wheels"
LOCK_FILE="${BASE_DIR}/requirements-lock.txt"

HDF5_VERSION="1.14.3"
HDF5_PREFIX="${BASE_DIR}/hdf5-${HDF5_VERSION}-openmpi5"
HDF5_BUILD_ROOT="${BASE_DIR}/build-hdf5-${HDF5_VERSION}"

PYTORCH_VERSION="2.5.1"
PYTORCH_SOURCE="${SOURCE_CACHE}/pytorch-${PYTORCH_VERSION}"

# Basic safety checks.
if [[ -z "${BASE_DIR}" || "${BASE_DIR}" == "/" ]]; then
    echo "ERROR: Refusing to clean an unsafe BASE_DIR: ${BASE_DIR}" >&2
    exit 1
fi

if [[ ! -f "${BASE_DIR}/submit_install_1.sh" ||
      ! -f "${BASE_DIR}/submit_install_2.sh" ]]; then
    echo "ERROR: Installation scripts were not found in ${BASE_DIR}." >&2
    exit 1
fi

echo "The following installation artifacts will be removed:"
printf '  %s\n' \
    "${VENV_DIR}" \
    "${HDF5_PREFIX}" \
    "${HDF5_BUILD_ROOT}" \
    "${PYTORCH_SOURCE}" \
    "${WHEEL_DIR}" \
    "${LOCK_FILE}"

# Remove the installation products.
rm -rf -- \
    "${VENV_DIR}" \
    "${HDF5_PREFIX}" \
    "${HDF5_BUILD_ROOT}" \
    "${PYTORCH_SOURCE}" \
    "${WHEEL_DIR}"

rm -f -- "${LOCK_FILE}"

# Preserve downloaded source archives, including the HDF5 tarball.
# Remove SOURCE_CACHE only if it is empty after removing PyTorch.
if [[ -d "${SOURCE_CACHE}" ]]; then
    rmdir "${SOURCE_CACHE}" 2>/dev/null || true
fi

echo
echo "Installation artifacts removed successfully."
echo "Preserved any remaining downloaded source archives in:"
echo "  ${SOURCE_CACHE}"
