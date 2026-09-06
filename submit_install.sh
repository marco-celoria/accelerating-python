#!/bin/bash -l
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --partition=lrd_all_serial
#SBATCH --time=4:00:00
#SBATCH --mem=30GB
#SBATCH --job-name=job_install
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Before submission, run on the login node:
#   mkdir -p logs
#   sbatch submit_install.sh

set -euo pipefail


# ---------------------------------------------------------------------------
# User configuration
# ---------------------------------------------------------------------------

BASE_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
VENV_DIR="${BASE_DIR}/pygpu_cuda122"
SOURCE_CACHE="${BASE_DIR}/sources"
WHEEL_DIR="${BASE_DIR}/wheels"
CUPY_CACHE_DIR="${BASE_DIR}/.cupy/kernel_cache"
LOCK_FILE="${BASE_DIR}/requirements-lock.txt"

HDF5_VERSION="1.14.3"
HDF5_TAG="hdf5-1_14_3"
HDF5_PREFIX="${BASE_DIR}/hdf5-${HDF5_VERSION}-openmpi5"
HDF5_BUILD_ROOT="${BASE_DIR}/build-hdf5-${HDF5_VERSION}"
HDF5_TARBALL="${SOURCE_CACHE}/hdf5-${HDF5_VERSION}.tar.gz"

PYTORCH_VERSION="2.5.1"
PYTORCH_SOURCE="${SOURCE_CACHE}/pytorch-${PYTORCH_VERSION}"

BUILD_JOBS="${SLURM_CPUS_PER_TASK:-4}"

export OMP_NUM_THREADS="${BUILD_JOBS}"
export MAX_JOBS="${BUILD_JOBS}"
export MAKEFLAGS="-j${BUILD_JOBS}"

mkdir -p "${SOURCE_CACHE}" "${WHEEL_DIR}" "${CUPY_CACHE_DIR}"


# ---------------------------------------------------------------------------
# Leonardo modules
# ---------------------------------------------------------------------------

module purge
module load gcc/12.2.0-binut2.41
module load cmake/3.27.9
module load cuda/12.2
module load python/3.11.7
module load openmpi/5.0.9--gcc--12.2.0-cuda-12.2
module load nccl/2.22.3-1--gcc--12.2.0-cuda-12.2-spack0.22
module load gdrcopy/2.4.1-cuda-12.2
module load cudnn/8.9.7.29-12--gcc--12.2.0-cuda-12.2 
module list


# ---------------------------------------------------------------------------
# CUDA 12.2 environment
# ---------------------------------------------------------------------------

NVCC_PATH="$(readlink -f "$(command -v nvcc)")"
CUDA_ROOT="$(dirname "$(dirname "${NVCC_PATH}")")"

export CUDA_HOME="${CUDA_ROOT}"
export CUDA_PATH="${CUDA_ROOT}"
export CUDACXX="${CUDA_ROOT}/bin/nvcc"

export PATH="${CUDA_ROOT}/bin:${PATH}"
export CPATH="${CUDA_ROOT}/include${CPATH:+:${CPATH}}"
export LIBRARY_PATH="${CUDA_ROOT}/lib64:${CUDA_ROOT}/targets/x86_64-linux/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export LD_LIBRARY_PATH="${CUDA_ROOT}/lib64:${CUDA_ROOT}/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

export CUPY_CACHE_DIR
export CUPY_ACCELERATORS="cub"


echo "===== Toolchain ====="
echo "BASE_DIR=${BASE_DIR}"
echo "Python=$(command -v python)"
echo "gcc=$(command -v gcc)"
echo "g++=$(command -v g++)"
echo "mpicc=$(command -v mpicc)"
echo "mpicxx=$(command -v mpicxx)"
echo "mpifort=$(command -v mpifort)"
echo "nvcc=$(command -v nvcc)"
echo "CUDA_HOME=${CUDA_HOME}"
echo "CUDA_PATH=${CUDA_PATH}"
echo "CUDACXX=${CUDACXX}"

python --version
gcc -dumpfullversion -dumpversion
mpicc --show
nvcc --version

# Verify the CUDA development toolkit.
test -x "${CUDA_HOME}/bin/nvcc"
test -f "${CUDA_HOME}/include/cuda.h"
test -f "${CUDA_HOME}/include/cooperative_groups.h"

if [[ -f "${CUDA_HOME}/lib64/libcudadevrt.a" ]]; then
    CUDADEVRT_PATH="${CUDA_HOME}/lib64/libcudadevrt.a"
elif [[ -f "${CUDA_HOME}/targets/x86_64-linux/lib/libcudadevrt.a" ]]; then
    CUDADEVRT_PATH="${CUDA_HOME}/targets/x86_64-linux/lib/libcudadevrt.a"
else
    echo "ERROR: libcudadevrt.a was not found under ${CUDA_HOME}." >&2
    exit 1
fi

echo "CUDA device runtime: ${CUDADEVRT_PATH}"


# ---------------------------------------------------------------------------
# Locate the NCCL installation exposed by the Leonardo module
# ---------------------------------------------------------------------------

NCCL_LIBRARY=""

while IFS= read -r directory; do
    [[ -z "${directory}" ]] && continue

    if [[ -e "${directory}/libnccl.so" ]]; then
        NCCL_LIBRARY="${directory}/libnccl.so"
        break
    fi

    if [[ -e "${directory}/libnccl.so.2" ]]; then
        NCCL_LIBRARY="${directory}/libnccl.so.2"
        break
    fi

done < <(
    printf '%s' "${LD_LIBRARY_PATH:-}" |
    tr ':' '\n'
)

if [[ -z "${NCCL_LIBRARY}" ]]; then
    echo "ERROR: libnccl.so was not found after loading the NCCL module." >&2
    echo "Inspect the module with:" >&2
    echo "  module show nccl/2.22.3-1--gcc--12.2.0-cuda-12.2-spack0.22" >&2
    exit 1
fi

NCCL_LIB_DIR="$(dirname "${NCCL_LIBRARY}")"
NCCL_ROOT="$(dirname "${NCCL_LIB_DIR}")"
NCCL_INCLUDE_DIR="${NCCL_ROOT}/include"

if [[ ! -f "${NCCL_INCLUDE_DIR}/nccl.h" ]]; then
    echo "ERROR: ${NCCL_INCLUDE_DIR}/nccl.h was not found." >&2
    exit 1
fi


export NCCL_ROOT
export NCCL_INCLUDE_DIR
export NCCL_LIB_DIR
export NCCL_LIBRARY
export NCCL_INCLUDE_DIRS="${NCCL_INCLUDE_DIR}"
export NCCL_LIBRARIES="${NCCL_LIBRARY}"

export CMAKE_PREFIX_PATH="${NCCL_ROOT}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"

echo "===== System NCCL ====="
echo "NCCL_ROOT=${NCCL_ROOT}"
echo "NCCL_INCLUDE_DIR=${NCCL_INCLUDE_DIR}"
echo "NCCL_LIB_DIR=${NCCL_LIB_DIR}"
echo "NCCL_LIBRARY=${NCCL_LIBRARY}"

# ---------------------------------------------------------------------------
# Locate the cuDNN installation exposed by the Leonardo module
# ---------------------------------------------------------------------------

CUDNN_HEADER=""

while IFS= read -r directory; do
    [[ -n "${directory}" ]] || continue
    [[ -d "${directory}" ]] || continue

    if [[ -f "${directory}/cudnn.h" ]]; then
        CUDNN_HEADER="${directory}/cudnn.h"
        break
    fi
done < <(
    printf '%s' "${CPATH:-}" |
        tr ':' '\n'
)

CUDNN_LIBRARY=""

while IFS= read -r directory; do
    [[ -n "${directory}" ]] || continue
    [[ -d "${directory}" ]] || continue

    # Use -e because libcudnn.so is commonly a symbolic link.
    if [[ -e "${directory}/libcudnn.so" ]]; then
        CUDNN_LIBRARY="${directory}/libcudnn.so"
        break
    fi

    if [[ -e "${directory}/libcudnn.so.8" ]]; then
        CUDNN_LIBRARY="${directory}/libcudnn.so.8"
        break
    fi
done < <(
    printf '%s' "${LD_LIBRARY_PATH:-}" |
        tr ':' '\n'
)

if [[ -z "${CUDNN_HEADER}" ]]; then
    echo "ERROR: cudnn.h was not found after loading the cuDNN module." >&2
    echo "CPATH entries:" >&2
    printf '%s' "${CPATH:-}" |
        tr ':' '\n' >&2
    exit 1
fi

if [[ -z "${CUDNN_LIBRARY}" ]]; then
    echo "ERROR: libcudnn.so or libcudnn.so.8 was not found." >&2
    echo "LD_LIBRARY_PATH entries:" >&2
    printf '%s' "${LD_LIBRARY_PATH:-}" |
        tr ':' '\n' >&2
    exit 1
fi

CUDNN_INCLUDE_DIR="$(dirname "${CUDNN_HEADER}")"
CUDNN_LIB_DIR="$(dirname "${CUDNN_LIBRARY}")"
CUDNN_ROOT="$(dirname "${CUDNN_LIB_DIR}")"

# Resolve the library link only for diagnostics. Keep CUDNN_LIBRARY pointing
# to the unversioned or ABI-versioned link exposed by the module.
CUDNN_LIBRARY_REAL="$(readlink -f "${CUDNN_LIBRARY}")"

if [[ ! -f "${CUDNN_INCLUDE_DIR}/cudnn.h" ]]; then
    echo "ERROR: Invalid cuDNN include directory: ${CUDNN_INCLUDE_DIR}" >&2
    exit 1
fi

if [[ ! -f "${CUDNN_LIBRARY_REAL}" ]]; then
    echo "ERROR: Invalid cuDNN library: ${CUDNN_LIBRARY}" >&2
    exit 1
fi

export CUDNN_ROOT
export CUDNN_INCLUDE_DIR
export CUDNN_LIB_DIR
export CUDNN_LIBRARY
export CUDNN_LIBRARY_REAL
# Additional variable names recognized by common CMake and PyTorch
# cuDNN-discovery paths.

export CUDNN_INCLUDE_PATH="${CUDNN_INCLUDE_DIR}"
export CUDNN_LIBRARY_PATH="${CUDNN_LIB_DIR}"
# Make the cuDNN prefix available to CMake without discarding existing
# module-provided prefixes.

export CMAKE_PREFIX_PATH="${CUDNN_ROOT}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"

echo "===== System cuDNN ====="
echo "CUDNN_ROOT=${CUDNN_ROOT}"
echo "CUDNN_INCLUDE_DIR=${CUDNN_INCLUDE_DIR}"
echo "CUDNN_LIB_DIR=${CUDNN_LIB_DIR}"
echo "CUDNN_LIBRARY=${CUDNN_LIBRARY}"
echo "CUDNN_LIBRARY_REAL=${CUDNN_LIBRARY_REAL}"
echo "CUDNN_INCLUDE_PATH=${CUDNN_INCLUDE_PATH}"
echo "CUDNN_LIBRARY_PATH=${CUDNN_LIBRARY_PATH}"


# ---------------------------------------------------------------------------
# Download and build parallel HDF5
# ---------------------------------------------------------------------------

if [[ ! -f "${HDF5_TARBALL}" ]]; then
    echo "Downloading HDF5 ${HDF5_VERSION}"

    wget \
        -O "${HDF5_TARBALL}.tmp" \
        "https://github.com/HDFGroup/hdf5/archive/refs/tags/${HDF5_TAG}.tar.gz"

    mv "${HDF5_TARBALL}.tmp" "${HDF5_TARBALL}"
fi

rm -rf "${HDF5_BUILD_ROOT}"

HDF5_SOURCE="${HDF5_BUILD_ROOT}/src"
HDF5_BUILD="${HDF5_BUILD_ROOT}/build"

mkdir -p "${HDF5_SOURCE}" "${HDF5_BUILD}"

tar \
    -xf "${HDF5_TARBALL}" \
    -C "${HDF5_SOURCE}" \
    --strip-components=1

cmake \
    -S "${HDF5_SOURCE}" \
    -B "${HDF5_BUILD}" \
    -DCMAKE_INSTALL_PREFIX="${HDF5_PREFIX}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="$(command -v mpicc)" \
    -DCMAKE_CXX_COMPILER="$(command -v mpicxx)" \
    -DCMAKE_Fortran_COMPILER="$(command -v mpifort)" \
    -DHDF5_ENABLE_PARALLEL=ON \
    -DHDF5_BUILD_FORTRAN=ON \
    -DHDF5_BUILD_HL_LIB=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_STATIC_LIBS=OFF

cmake \
    --build "${HDF5_BUILD}" \
    --parallel "${BUILD_JOBS}"

cmake --install "${HDF5_BUILD}"


echo "===== HDF5 configuration ====="
"${HDF5_PREFIX}/bin/h5pcc" -showconfig

export HDF5_HOME="${HDF5_PREFIX}"
export HDF5_DIR="${HDF5_PREFIX}"
export PATH="${HDF5_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${HDF5_HOME}/lib:${HDF5_HOME}/lib64:${LD_LIBRARY_PATH}"
export CMAKE_PREFIX_PATH="${HDF5_HOME}:${CMAKE_PREFIX_PATH}"


# ---------------------------------------------------------------------------
# Create one unified Python environment
# ---------------------------------------------------------------------------

rm -rf "${VENV_DIR}"
python -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel

# PyTorch source-build dependencies and the scientific stack.
python -m pip install \
    "ninja==1.13.0" \
    "pyyaml" \
    "typing-extensions" \
    "sympy==1.13.1" \
    "networkx" \
    "jinja2" \
    "fsspec" \
    "filelock" \
    "numpy==2.4.6" \
    "scipy==1.17.0" \
    "matplotlib==3.10.9"

# CuPy, Numba CUDA, and the CUDA discovery versions from the working stack.
python -m pip install \
    "cupy-cuda12x==14.1.1" \
    "numba==0.66.0" \
    "llvmlite==0.48.0" \
    "numba-cuda==0.30.4" \
    "cuda-bindings==12.9.7" \
    "cuda-core==0.6.0" \
    "cuda-pathfinder==1.8.0" \
    "cutensor-cu12==2.3.0"

MPI4PY_BUILD_MPICC="$(command -v mpicc)" \
    python -m pip install \
    --no-binary=mpi4py \
    --no-cache-dir \
    "mpi4py==4.1.2"

HDF5_MPI="ON" \
CC="$(command -v mpicc)" \
HDF5_DIR="${HDF5_HOME}" \
    python -m pip install \
    --no-binary=h5py \
    --no-cache-dir \
    "h5py==3.16.0"


# ---------------------------------------------------------------------------
# Download PyTorch 2.5.1 source
# ---------------------------------------------------------------------------

#if [[ ! -d "${PYTORCH_SOURCE}/.git" ]]; then
#    rm -rf "${PYTORCH_SOURCE}"
#
#    git clone \
#        --branch "v${PYTORCH_VERSION}" \
#        --depth 1 \
#        --recursive \
#        https://github.com/pytorch/pytorch.git \
#        "${PYTORCH_SOURCE}"
#else
#    git -C "${PYTORCH_SOURCE}" fetch \
#        --depth 1 \
#        origin \
#        "v${PYTORCH_VERSION}"
#
#    git -C "${PYTORCH_SOURCE}" checkout --force "v${PYTORCH_VERSION}"
#    git -C "${PYTORCH_SOURCE}" submodule sync --recursive
#    git -C "${PYTORCH_SOURCE}" submodule update --init --recursive
#fi

if [[ ! -d "${PYTORCH_SOURCE}/.git" ]]; then
    rm -rf "${PYTORCH_SOURCE}"

    git clone \
        --branch "v${PYTORCH_VERSION}" \
        --recursive \
        https://github.com/pytorch/pytorch.git \
        "${PYTORCH_SOURCE}"
else
    git -C "${PYTORCH_SOURCE}" fetch \
        --tags \
        origin

    git -C "${PYTORCH_SOURCE}" checkout \
        --force \
        "v${PYTORCH_VERSION}"

    git -C "${PYTORCH_SOURCE}" submodule sync \
        --recursive

    git -C "${PYTORCH_SOURCE}" submodule update \
        --init \
        --recursive \
        --jobs "${BUILD_JOBS}"
fi

export PYTORCH_BUILD_VERSION="${PYTORCH_VERSION}"
export PYTORCH_BUILD_NUMBER=1
echo "PYTORCH_BUILD_VERSION=${PYTORCH_BUILD_VERSION}"
echo "PYTORCH_BUILD_NUMBER=${PYTORCH_BUILD_NUMBER}"

# ---------------------------------------------------------------------------
# Configure PyTorch to use Leonardo CUDA and NCCL
# ---------------------------------------------------------------------------

export USE_CUDA=1
export USE_NCCL=1
export USE_SYSTEM_NCCL=1
export USE_DISTRIBUTED=1
export USE_MPI=1

# Build against the loaded Leonardo cuDNN module.
export USE_CUDNN=1
export USE_CUSPARSELT=0
export USE_MAGMA=0

# Leonardo Booster uses NVIDIA A100 GPUs, compute capability 8.0.
export TORCH_CUDA_ARCH_LIST="8.0"

export CC="$(command -v gcc)"
export CXX="$(command -v g++)"
export CMAKE_GENERATOR="Ninja"
export BUILD_TEST=0
export BUILD_BINARY=0
export USE_NUMPY=1
export USE_OPENMP=1
export USE_GLOO=1

# Ensure CMake can discover the virtual environment, HDF5, and system NCCL.
export CMAKE_PREFIX_PATH="${VENV_DIR}:${HDF5_HOME}:${NCCL_ROOT}:${CUDNN_ROOT}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"

echo "===== PyTorch build configuration ====="
echo "PYTORCH_SOURCE=${PYTORCH_SOURCE}"
echo "USE_CUDA=${USE_CUDA}"
echo "USE_NCCL=${USE_NCCL}"
echo "USE_SYSTEM_NCCL=${USE_SYSTEM_NCCL}"
echo "USE_MPI=${USE_MPI}"
echo "USE_CUDNN=${USE_CUDNN}"
echo "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
echo "MAX_JOBS=${MAX_JOBS}"
echo "CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}"


# ---------------------------------------------------------------------------
# Build the PyTorch wheel
# ---------------------------------------------------------------------------
cd "${PYTORCH_SOURCE}"

rm -rf build dist
rm -f "${WHEEL_DIR}"/torch-*.whl

export CFLAGS="${CFLAGS:-} -Wno-error=maybe-uninitialized -Wno-error=uninitialized"
export CXXFLAGS="${CXXFLAGS:-} -Wno-error=maybe-uninitialized -Wno-error=uninitialized"
echo "CFLAGS=${CFLAGS}"
echo "CXXFLAGS=${CXXFLAGS}"

export MAX_JOBS="${BUILD_JOBS}"

python -m pip wheel \
    --no-build-isolation \
    --no-deps \
    --no-cache-dir \
    --wheel-dir "${WHEEL_DIR}" \
    -v \
    .

TORCH_WHEEL="$(
    find "${WHEEL_DIR}" \
        -maxdepth 1 \
        -type f \
        -name 'torch-*.whl' \
        -print |
    sort |
    tail -n 1
)"

if [[ -z "${TORCH_WHEEL}" || ! -f "${TORCH_WHEEL}" ]]; then
    echo "ERROR: PyTorch wheel was not created in ${WHEEL_DIR}." >&2
    exit 1
fi

echo "Installing PyTorch wheel: ${TORCH_WHEEL}"

python -m pip install --no-deps "${TORCH_WHEEL}"

cd "${BASE_DIR}"

python -m pip install \
    --force-reinstall \
    --no-deps \
    "cuda-bindings==12.9.7" \
    "cuda-core==0.6.0" 

echo "===== PyTorch wheel requirements ====="

python - <<'PY'
from importlib.metadata import requires

requirements = requires("torch") or []

if requirements:
    for requirement in requirements:
        print(requirement)
else:
    print("No requirements recorded in the Torch package metadata.")
PY

# ---------------------------------------------------------------------------
# Validate package metadata
# ---------------------------------------------------------------------------


echo "===== Package consistency ====="

if ! python -m pip check; then
    echo "ERROR: Python package metadata is inconsistent." >&2
    echo "Inspect the Torch requirements printed above." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Validate imports, CUDA discovery, HDF5, and NCCL linkage
# ---------------------------------------------------------------------------

python - <<'PY'
import ctypes
import os
from pathlib import Path

import cupy as cp
import h5py
import numpy as np
import scipy
import torch
from mpi4py import MPI

print("===== Installed versions =====")
print("NumPy:", np.__version__)
print("SciPy:", scipy.__version__)
print("CuPy:", cp.__version__)
print("PyTorch:", torch.__version__)
print("PyTorch CUDA build:", torch.version.cuda)
print()
print("===== PyTorch build features =====")
print(torch.__config__.show())
print("Distributed available:", torch.distributed.is_available())
print("NCCL available:", torch.distributed.is_nccl_available())
print("MPI available:", torch.distributed.is_mpi_available())
print("Gloo available:", torch.distributed.is_gloo_available())
print("cuDNN version:", torch.backends.cudnn.version())

if torch.version.cuda != "12.2":
    raise RuntimeError(
        f"Expected PyTorch CUDA 12.2, got {torch.version.cuda!r}"
    )

if not torch.distributed.is_available():
    raise RuntimeError("PyTorch distributed support is unavailable")

if not torch.distributed.is_nccl_available():
    raise RuntimeError("PyTorch was built without NCCL support")

if not torch.distributed.is_mpi_available():
    raise RuntimeError("PyTorch was built without MPI support")

if torch.backends.cudnn.version() is None:
    raise RuntimeError("PyTorch was built without cuDNN support")

print("h5py:", h5py.__version__)
print("h5py MPI support:", h5py.get_config().mpi)
print("MPI vendor:", MPI.get_vendor())
print("MPI version:", MPI.Get_version())

print()
print("===== CUDA and NCCL environment =====")
print("CUDA_HOME:", os.environ.get("CUDA_HOME"))
print("CUDA_PATH:", os.environ.get("CUDA_PATH"))
print("NCCL_ROOT:", os.environ.get("NCCL_ROOT"))
print("NCCL_INCLUDE_DIR:", os.environ.get("NCCL_INCLUDE_DIR"))
print("NCCL_LIB_DIR:", os.environ.get("NCCL_LIB_DIR"))

cuda_root_text = cp.cuda.get_cuda_path()

print("CuPy CUDA root:", cuda_root_text)
print("PyTorch CUDA available on this node:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("PyTorch GPU:", torch.cuda.get_device_name(0))
else:
    print("No GPU visible; runtime GPU validation is deferred.")


if not h5py.get_config().mpi:
    raise RuntimeError("h5py was built without MPI support")

cuda_root_text = cp.cuda.get_cuda_path()
if cuda_root_text is None:
    raise RuntimeError(
        "CuPy did not detect the Leonardo CUDA toolkit. "
        "Cooperative RawKernel compilation may fail."
    )

cuda_root = Path(cuda_root_text)
cudadevrt_candidates = (
    cuda_root / "lib64" / "libcudadevrt.a",
    cuda_root / "targets" / "x86_64-linux" / "lib" / "libcudadevrt.a",
)

cudadevrt = next(
    (candidate for candidate in cudadevrt_candidates if candidate.is_file()),
    None,
)

if cudadevrt is None:
    raise RuntimeError(
        f"CuPy detected CUDA at {cuda_root}, but libcudadevrt.a was not found"
    )

print("libcudadevrt:", cudadevrt)

# Load the module-provided NCCL library and query its version directly.

nccl_library = os.environ.get("NCCL_LIBRARY")

if not nccl_library:
    raise RuntimeError("NCCL_LIBRARY is not set")

print("Loading NCCL library:", nccl_library)

nccl = ctypes.CDLL(nccl_library)
nccl.ncclGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
nccl.ncclGetVersion.restype = ctypes.c_int

version = ctypes.c_int()
status = nccl.ncclGetVersion(ctypes.byref(version))

if status != 0:
    raise RuntimeError(f"ncclGetVersion failed with status {status}")

print("Loaded NCCL integer version:", version.value)

if version.value != 22203:
    raise RuntimeError(
        f"Expected NCCL 2.22.3 (22203), but loaded version {version.value}"
    )

print("Installation-node validation passed.")
PY


# ---------------------------------------------------------------------------
# Verify that libtorch_cuda resolves NCCL from the loaded module
# ---------------------------------------------------------------------------

TORCH_CUDA_LIBRARY="$(python - <<'PY'
from pathlib import Path
import torch

root = Path(torch.__file__).resolve().parent
matches = sorted(root.rglob("libtorch_cuda.so"))

if not matches:
    raise SystemExit("libtorch_cuda.so was not found")

print(matches[0])
PY
)"


echo "===== PyTorch CUDA library ====="
echo "${TORCH_CUDA_LIBRARY}"

echo "===== NCCL linkage ====="
NCCL_LINKAGE="$(ldd "${TORCH_CUDA_LIBRARY}" | grep -i 'libnccl' || true)"

if [[ -z "${NCCL_LINKAGE}" ]]; then
    echo "ERROR: libtorch_cuda.so does not show a dynamic NCCL dependency." >&2
    exit 1
fi

printf '%s\n' "${NCCL_LINKAGE}"

if printf '%s\n' "${NCCL_LINKAGE}" | grep -q 'site-packages/nvidia/nccl'; then
    echo "ERROR: PyTorch resolved NCCL from a Python package, not the module." >&2
    exit 1
fi


# ---------------------------------------------------------------------------
# Save exact environment
# ---------------------------------------------------------------------------

python -m pip freeze > "${LOCK_FILE}"


echo
echo "===== INSTALLATION COMPLETED ====="
echo "HDF5:          ${HDF5_PREFIX}"
echo "Virtualenv:    ${VENV_DIR}"
echo "PyTorch wheel: ${TORCH_WHEEL}"
echo "CUDA:          ${CUDA_HOME}"
echo "NCCL:          ${NCCL_LIBRARY}"
echo "Lock file:     ${LOCK_FILE}"
echo
echo "Run a separate GPU validation job before production use."
