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

# NOTE: Ensure 'mkdir -p logs' is run on the login node BEFORE calling 'sbatch submit_install.sh'
mkdir -p logs

set -euo pipefail

# USER CONFIG
BASE_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
HDF5_VERSION="1.14.3"
HDF5_TAG="hdf5-1_14_3"
INSTALL_PREFIX="${BASE_DIR}/hdf5-${HDF5_VERSION}-openmpi5"
BUILD_ROOT="${BASE_DIR}/build-hdf5-${HDF5_VERSION}"
SOURCE_CACHE="${BASE_DIR}/sources"
HDF5_TARBALL="${SOURCE_CACHE}/hdf5-${HDF5_VERSION}.tar.gz"
VENV_DIR="${BASE_DIR}/pygpu_cuda122"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MAKEFLAGS="-j${SLURM_CPUS_PER_TASK:-4}"

module purge
module load gcc/12.2.0-binut2.41
module load cmake/3.27.9
module load cuda/12.2
module load python/3.11.7
module load openmpi/5.0.9--gcc--12.2.0-cuda-12.2
module load nccl/2.22.3-1--gcc--12.2.0-cuda-12.2-spack0.22
module load gdrcopy/2.4.1-cuda-12.2

module list
echo "Python: $(command -v python)"
echo "mpicc:  $(command -v mpicc)"
echo "mpicxx: $(command -v mpicxx)"
echo "nvcc:   $(command -v nvcc)"
python --version
mpicc --show
nvcc --version

export CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
export CUDA_PATH="${CUDA_HOME}"
echo "CUDA_HOME=${CUDA_HOME}"

# Safe header verification (won't abort script under set -e)
if [[ -f "${CUDA_HOME}/include/mma.h" || -f "${CUDA_HOME}/include/crt/mma.h" ]]; then
    echo "CUDA MMA headers found."
else
    echo "Warning: mma.h not found in standard paths."
fi

# Download HDF5 source
mkdir -p "${SOURCE_CACHE}"
if [[ ! -f "${HDF5_TARBALL}" ]]; then
  echo "Downloading HDF5 ${HDF5_VERSION}"
  wget -O "${HDF5_TARBALL}.tmp" "https://github.com/HDFGroup/hdf5/archive/refs/tags/${HDF5_TAG}.tar.gz"
  mv "${HDF5_TARBALL}.tmp" "${HDF5_TARBALL}"
fi

# Extract source safely
rm -rf "${BUILD_ROOT}"
HDF5_SOURCE="${BUILD_ROOT}/src"
HDF5_BUILD="${BUILD_ROOT}/build"
mkdir -p "${HDF5_SOURCE}" "${HDF5_BUILD}"
tar -xf "${HDF5_TARBALL}" -C "${HDF5_SOURCE}" --strip-components=1

# Configure and Build HDF5
cmake \
  -S "${HDF5_SOURCE}" \
  -B "${HDF5_BUILD}" \
  -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$(command -v mpicc)" \
  -DCMAKE_CXX_COMPILER="$(command -v mpicxx)" \
  -DCMAKE_Fortran_COMPILER="$(command -v mpifort)" \
  -DHDF5_ENABLE_PARALLEL=ON \
  -DHDF5_BUILD_FORTRAN=ON \
  -DHDF5_BUILD_HL_LIB=ON \
  -DBUILD_SHARED_LIBS=ON \
  -DBUILD_STATIC_LIBS=OFF

cmake --build "${HDF5_BUILD}" --parallel "${SLURM_CPUS_PER_TASK:-4}"
cmake --install "${HDF5_BUILD}"

echo "HDF5 configuration:"
"${INSTALL_PREFIX}/bin/h5pcc" -showconfig

echo "Verifying installation..."
${INSTALL_PREFIX}/bin/h5pcc -showconfig | grep -i parallel || true
echo ""

# Export environment paths (Including lib64 for CMake installs)
export HDF5_HOME="${INSTALL_PREFIX}"
export HDF5_DIR="${INSTALL_PREFIX}"
export PATH="${HDF5_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${HDF5_HOME}/lib:${HDF5_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CUPY_ACCELERATORS="cub,cutensor"

# Setup Python Virtual Environment
rm -rf "${VENV_DIR}"
python -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel

python -m pip install \
  "numpy==2.4.6" \
  "scipy==1.17.0" \
  "numba==0.66.0" \
  "llvmlite==0.48.0" \
  "numba-cuda==0.30.4" \
  "cuda-bindings>=12.9.1,<13" \
  "cuda-core==0.6.0" \
  "cuda-pathfinder>=1.4,<2" \
  "matplotlib==3.10.9" \
  "cupy-cuda12x==14.1.1" \
  "cutensor-cu12==2.3.0"


MPI4PY_BUILD_MPICC="$(command -v mpicc)" python -m pip install --no-binary=mpi4py --no-cache-dir "mpi4py==4.1.2"

HDF5_MPI="ON" CC="$(command -v mpicc)" HDF5_DIR="${HDF5_HOME}" \
    python -m pip install --no-binary=h5py --no-cache-dir "h5py==3.16.0"

#python -m pip install "cutensor-cu12==2.3.0"

# Validate Python environment
python -m pip check
python - <<'PY'
import os
import h5py
import numpy as np
import scipy
import cupy as cp
from mpi4py import MPI

print("NumPy:", np.__version__)
print("SciPy:", scipy.__version__)
print("h5py:", h5py.__version__)
print("h5py MPI support:", h5py.get_config().mpi)
print("MPI vendor:", MPI.get_vendor())
print("MPI version:", MPI.Get_version())
print("MPI library:\n", MPI.Get_library_version())
print("CUDA_HOME:", os.environ.get("CUDA_HOME"))
if not h5py.get_config().mpi:
    raise RuntimeError("h5py was built without MPI support")
PY

# Save exact environment
python -m pip freeze > "${BASE_DIR}/requirements-lock.txt"

echo
echo "===== DONE ====="
echo "HDF5: ${INSTALL_PREFIX}"
echo "Virtualenv: ${VENV_DIR}"
