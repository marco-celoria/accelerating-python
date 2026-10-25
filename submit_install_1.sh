#!/bin/bash -l
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --partition=lrd_all_serial
#SBATCH --time=4:00:00
#SBATCH --mem=30GB
#SBATCH --job-name=run_install_1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Before submission, run on the login node:
#   mkdir -p logs
#   job1=$(sbatch --parsable submit_install_1.sh)
#   echo "Submitted stage 1 as job ${job1}"
#   job2=$(sbatch --parsable --dependency=afterok:"${job1}" submit_install_2.sh)
#   echo "Submitted stage 2 as job ${job2}"

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
# Select the GCC 12.2 runtime libraries
# ---------------------------------------------------------------------------

GCC_LIBSTDCPP="$(g++ -print-file-name=libstdc++.so.6)"
GCC_LIBGCC="$(gcc -print-file-name=libgcc_s.so.1)"

if [[ "${GCC_LIBSTDCPP}" == "libstdc++.so.6" ||
      ! -f "${GCC_LIBSTDCPP}" ]]; then
    echo "ERROR: The active GCC did not locate libstdc++.so.6." >&2
    exit 1
fi

if [[ "${GCC_LIBGCC}" == "libgcc_s.so.1" ||
      ! -f "${GCC_LIBGCC}" ]]; then
    echo "ERROR: The active GCC did not locate libgcc_s.so.1." >&2
    exit 1
fi

GCC_LIBSTDCPP_REAL="$(readlink -f "${GCC_LIBSTDCPP}")"
GCC_LIBGCC_REAL="$(readlink -f "${GCC_LIBGCC}")"
GCC_RUNTIME_DIR="$(dirname "${GCC_LIBSTDCPP_REAL}")"

# Do not use grep -q with pipefail because early termination may cause
# strings to receive SIGPIPE. Do not require an exact whole-line match
# because the strings output format can vary.
if ! strings "${GCC_LIBSTDCPP_REAL}" |
    grep -F 'GLIBCXX_3.4.30' >/dev/null; then
    echo "ERROR: ${GCC_LIBSTDCPP_REAL} lacks GLIBCXX_3.4.30." >&2
    exit 1
fi


export GCC_RUNTIME_DIR

# Force every child process to load the GCC 12.2 runtime before any
# Python extension or module dependency can load another libstdc++.
export LD_PRELOAD="${GCC_LIBSTDCPP_REAL}:${GCC_LIBGCC_REAL}${LD_PRELOAD:+:${LD_PRELOAD}}"

export LD_LIBRARY_PATH="${GCC_RUNTIME_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export LIBRARY_PATH="${GCC_RUNTIME_DIR}${LIBRARY_PATH:+:${LIBRARY_PATH}}"

echo "===== GCC runtime ====="
echo "CC=$(command -v gcc)"
echo "CXX=$(command -v g++)"
echo "libstdc++=${GCC_LIBSTDCPP_REAL}"
echo "libgcc_s=${GCC_LIBGCC_REAL}"
echo "GCC_RUNTIME_DIR=${GCC_RUNTIME_DIR}"
echo "LD_PRELOAD=${LD_PRELOAD}"

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

# ---------------------------------------------------------------------------
# Download PyTorch 2.5.1 source
# ---------------------------------------------------------------------------

if [[ ! -d "${PYTORCH_SOURCE}/.git" ]]; then
    rm -rf "${PYTORCH_SOURCE}"

    echo "Cloning PyTorch v${PYTORCH_VERSION}"

    git \
        -c http.version=HTTP/1.1 \
        clone \
        --branch "v${PYTORCH_VERSION}" \
        --depth 1 \
        --single-branch \
        --no-tags \
        https://github.com/pytorch/pytorch.git \
        "${PYTORCH_SOURCE}"

    git -C "${PYTORCH_SOURCE}" submodule sync --recursive

    git \
        -C "${PYTORCH_SOURCE}" \
        -c http.version=HTTP/1.1 \
        submodule update \
        --init \
        --recursive \
        --depth 1 \
        --jobs "${BUILD_JOBS}"
else
    git -C "${PYTORCH_SOURCE}" fetch \
        --depth 1 \
        origin \
        "refs/tags/v${PYTORCH_VERSION}:refs/tags/v${PYTORCH_VERSION}"

    git -C "${PYTORCH_SOURCE}" checkout \
        --force \
        "v${PYTORCH_VERSION}"

    git -C "${PYTORCH_SOURCE}" submodule sync \
        --recursive

    git \
        -C "${PYTORCH_SOURCE}" \
        -c http.version=HTTP/1.1 \
        submodule update \
        --init \
        --recursive \
        --depth 1 \
        --jobs "${BUILD_JOBS}"
fi

echo "===== PyTorch source revision ====="

git -C "${PYTORCH_SOURCE}" describe \
    --tags \
    --exact-match

git -C "${PYTORCH_SOURCE}" submodule status \
    --recursive

if [[ "$(
    git -C "${PYTORCH_SOURCE}" describe \
        --tags \
        --exact-match
)" != "v${PYTORCH_VERSION}" ]]; then
    echo "ERROR: PyTorch source is not at v${PYTORCH_VERSION}." >&2
    exit 1
fi

echo "===== Preparing clean PyTorch build tree ====="

rm -rf \
    "${PYTORCH_SOURCE}/build" \
    "${PYTORCH_SOURCE}/dist"

rm -f "${WHEEL_DIR}"/torch-*.whl

# PyTorch 2.5.1 declares Triton 3.1.0 on Linux.
python -m pip install "triton==3.1.0"

echo
echo "===== INSTALLATION STAGE 1 COMPLETED ====="
echo "HDF5:       ${HDF5_PREFIX}"
echo "Virtualenv: ${VENV_DIR}"
echo "PyTorch:    ${PYTORCH_SOURCE}"
echo
echo "Stage 2 will start automatically after this stage succeeds."
