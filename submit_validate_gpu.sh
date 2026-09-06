#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=boost_usr_prod
#SBATCH --time=0:10:00
#SBATCH --mem=16GB
#SBATCH --job-name=validate_gpu_stack
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --qos=boost_qos_dbg

set -euo pipefail

# ---------------------------------------------------------------------------
# User configuration
# ---------------------------------------------------------------------------

BASE_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
VENV_DIR="${BASE_DIR}/pygpu_cuda122"
CUPY_CACHE_DIR="${BASE_DIR}/.cupy/kernel_cache"
HDF5_HOME="${BASE_DIR}/hdf5-1.14.3-openmpi5"

mkdir -p "${BASE_DIR}/logs" "${CUPY_CACHE_DIR}"

[[ -x "${VENV_DIR}/bin/python" ]] || {
    echo "ERROR: Virtual environment not found: ${VENV_DIR}" >&2
    exit 1
}

[[ -x "${HDF5_HOME}/bin/h5pcc" ]] || {
    echo "ERROR: Parallel HDF5 installation not found: ${HDF5_HOME}" >&2
    exit 1
}

# ---------------------------------------------------------------------------
# Leonardo modules
# ---------------------------------------------------------------------------

module purge
module load gcc/12.2.0-binut2.41
module load cuda/12.2
module load python/3.11.7
module load openmpi/5.0.9--gcc--12.2.0-cuda-12.2
module load nccl/2.22.3-1--gcc--12.2.0-cuda-12.2-spack0.22
module load gdrcopy/2.4.1-cuda-12.2
module load cudnn/8.9.7.29-12--gcc--12.2.0-cuda-12.2
module list

export MPI4PY_RC_THREAD_LEVEL=funneled
export UCX_TLS=rc,cuda_copy,cuda_ipc,self
export UCX_MEMTYPE_CACHE=n
export OMPI_MCA_pml=ucx

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

# Do not use grep -q here. With pipefail, early grep termination can cause
# strings to receive SIGPIPE and make a successful match look like failure.
if ! strings "${GCC_LIBSTDCPP_REAL}" |
    grep -xF 'GLIBCXX_3.4.30' >/dev/null; then
    echo "ERROR: ${GCC_LIBSTDCPP_REAL} lacks GLIBCXX_3.4.30." >&2
    exit 1
fi

export GCC_RUNTIME_DIR
# Preload the GCC 12.2 C++ runtime before any Python extension can load
# another libstdc++.so.6 into the process.
export LD_PRELOAD="${GCC_LIBSTDCPP_REAL}:${GCC_LIBGCC_REAL}${LD_PRELOAD:+:${LD_PRELOAD}}"
export LD_LIBRARY_PATH="${GCC_RUNTIME_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export LIBRARY_PATH="${GCC_RUNTIME_DIR}${LIBRARY_PATH:+:${LIBRARY_PATH}}"

# ---------------------------------------------------------------------------
# CUDA, HDF5, and application environment
# ---------------------------------------------------------------------------

NVCC_PATH="$(readlink -f "$(command -v nvcc)")"
CUDA_ROOT="$(dirname "$(dirname "${NVCC_PATH}")")"

export CUDA_HOME="${CUDA_ROOT}"
export CUDA_PATH="${CUDA_ROOT}"
export CUDACXX="${CUDA_ROOT}/bin/nvcc"
export PATH="${HDF5_HOME}/bin:${CUDA_ROOT}/bin:${PATH}"
export CPATH="${CUDA_ROOT}/include${CPATH:+:${CPATH}}"
export LIBRARY_PATH="${GCC_RUNTIME_DIR}:${CUDA_ROOT}/lib64:${CUDA_ROOT}/targets/x86_64-linux/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export LD_LIBRARY_PATH="${GCC_RUNTIME_DIR}:${HDF5_HOME}/lib:${HDF5_HOME}/lib64:${CUDA_ROOT}/lib64:${CUDA_ROOT}/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export HDF5_HOME
export HDF5_DIR="${HDF5_HOME}"
export CUPY_CACHE_DIR
export CUPY_ACCELERATORS="cub"

source "${VENV_DIR}/bin/activate"

# ---------------------------------------------------------------------------
# Allocation and toolchain checks
# ---------------------------------------------------------------------------

[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || {
    echo "ERROR: Slurm did not assign any visible CUDA devices." >&2
    exit 1
}

command -v nvidia-smi >/dev/null 2>&1 || {
    echo "ERROR: nvidia-smi is unavailable on the allocated node." >&2
    exit 1
}

echo "===== Runtime environment ====="
echo "BASE_DIR=${BASE_DIR}"
echo "VENV_DIR=${VENV_DIR}"
echo "Python=$(command -v python)"
echo "CC=$(command -v gcc)"
echo "CXX=$(command -v g++)"
echo "NVCC=$(command -v nvcc)"
echo "CUDA_HOME=${CUDA_HOME}"
echo "HDF5_HOME=${HDF5_HOME}"
echo "libstdc++=${GCC_LIBSTDCPP_REAL}"
echo "libgcc_s=${GCC_LIBGCC_REAL}"
echo "GCC_RUNTIME_DIR=${GCC_RUNTIME_DIR}"
echo "CUPY_CACHE_DIR=${CUPY_CACHE_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-unset}"

python --version
nvcc --version
nvidia-smi

# ---------------------------------------------------------------------------
# Verify Torch resolves the selected GCC 12.2 runtime
# ---------------------------------------------------------------------------

TORCH_PYTHON_LIBRARY="$(
    python - <<'PY'
from pathlib import Path
import torch

root = Path(torch.__file__).resolve().parent
matches = sorted(root.rglob("libtorch_python.so"))
if not matches:
    raise SystemExit("libtorch_python.so was not found")
print(matches[0])
PY
)"

echo "===== Torch C++ runtime linkage ====="
echo "libtorch_python=${TORCH_PYTHON_LIBRARY}"

LIBSTDCPP_LINKAGE="$(
    ldd "${TORCH_PYTHON_LIBRARY}" |
        grep -F 'libstdc++.so.6' || true
)"

LIBGCC_LINKAGE="$(
    ldd "${TORCH_PYTHON_LIBRARY}" |
        grep -F 'libgcc_s.so.1' || true
)"

printf '%s\n' "${LIBSTDCPP_LINKAGE}"
printf '%s\n' "${LIBGCC_LINKAGE}"

[[ -n "${LIBSTDCPP_LINKAGE}" ]] || {
    echo "ERROR: Torch libstdc++ linkage was not found." >&2
    exit 1
}

[[ -n "${LIBGCC_LINKAGE}" ]] || {
    echo "ERROR: Torch libgcc_s linkage was not found." >&2
    exit 1
}

if [[ "${LIBSTDCPP_LINKAGE}" != *"${GCC_RUNTIME_DIR}/libstdc++.so.6"* ]]; then
    echo "ERROR: Torch is not using the selected GCC 12.2 libstdc++.so.6." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Python GPU-stack validation
# ---------------------------------------------------------------------------


python - <<'PY'
from __future__ import annotations

import platform
import time

import cupy as cp
import h5py
import numpy as np
import torch
from mpi4py import MPI
from numba import __version__ as numba_version
from numba import cuda

N = 1_000_000
THREADS = 256
BLOCKS = min((N + THREADS - 1) // THREADS, 4096)

print("\n" + "=" * 72)
print("GPU SOFTWARE STACK VALIDATION")
print("=" * 72)
print("Python:", platform.python_version())
print("NumPy:", np.__version__)
print("PyTorch:", torch.__version__)
print("CuPy:", cp.__version__)
print("Numba:", numba_version)
print("h5py:", h5py.__version__)

print("\n" + "=" * 72)
print("MPI AND PARALLEL HDF5 VALIDATION")
print("=" * 72)
print("h5py MPI support:", h5py.get_config().mpi)
print("MPI vendor:", MPI.get_vendor())
print("MPI standard version:", MPI.Get_version())
print("MPI world size:", MPI.COMM_WORLD.Get_size())
print("MPI rank:", MPI.COMM_WORLD.Get_rank())
if not h5py.get_config().mpi:
    raise RuntimeError("h5py was built without MPI support")
print("mpi4py and parallel h5py imports: PASS")

print("\n" + "=" * 72)
print("PYTORCH CUDA VALIDATION")
print("=" * 72)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Visible devices:", torch.cuda.device_count())

if torch.__version__.split("+", 1)[0] != "2.5.1":
    raise RuntimeError(f"Unexpected PyTorch version: {torch.__version__}")
if torch.version.cuda != "12.2":
    raise RuntimeError(f"Unexpected PyTorch CUDA build: {torch.version.cuda}")
if not torch.cuda.is_available():
    raise RuntimeError("PyTorch cannot access a CUDA GPU")
if torch.cuda.device_count() < 1:
    raise RuntimeError("PyTorch reports zero visible CUDA devices")

for index in range(torch.cuda.device_count()):
    print(f"GPU {index}:", torch.cuda.get_device_name(index))

torch.cuda.set_device(0)
x = torch.arange(N, dtype=torch.float64, device="cuda")
torch.cuda.synchronize()
t0 = time.perf_counter()
result = x.square().sum()
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0

expected_integer = (N - 1) * N * (2 * N - 1) // 6
expected = torch.tensor(float(expected_integer), dtype=torch.float64)
torch.testing.assert_close(result.cpu(), expected, rtol=1e-12, atol=1e-6)

cudnn_version = torch.backends.cudnn.version()
nccl_available = torch.distributed.is_nccl_available()
mpi_available = torch.distributed.is_mpi_available()

print("GPU computation result:", result.item())
print(f"GPU execution time: {elapsed:.6f} s")
print("cuDNN version:", cudnn_version)
print("Distributed available:", torch.distributed.is_available())
print("NCCL available:", nccl_available)
print("MPI available:", mpi_available)

if cudnn_version is None:
    raise RuntimeError("PyTorch was built without cuDNN support")
if cudnn_version != 8907:
    raise RuntimeError(
        f"Expected cuDNN 8.9.7, found integer version {cudnn_version}"
    )
if not torch.distributed.is_available():
    raise RuntimeError("PyTorch distributed support is unavailable")
if not nccl_available:
    raise RuntimeError("PyTorch was built without NCCL support")
if not mpi_available:
    raise RuntimeError("PyTorch was built without MPI support")

print("PyTorch CUDA execution: PASS")

print("\n" + "=" * 72)
print("CUPY RAWKERNEL VALIDATION")
print("=" * 72)
if cp.cuda.runtime.getDeviceCount() < 1:
    raise RuntimeError("CuPy reports zero visible CUDA devices")

print("CUDA root:", cp.cuda.get_cuda_path())
print("CUDA runtime version:", cp.cuda.runtime.runtimeGetVersion())
print("CUDA driver version:", cp.cuda.runtime.driverGetVersion())

cupy_source = r'''
extern "C" __global__
void affine_transform(const double* input, double* output, long long n)
{
    const long long start =
        (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long stride =
        (long long)blockDim.x * gridDim.x;

    for (long long i = start; i < n; i += stride) {
        output[i] = 2.0 * input[i] + 1.0;
    }
}
'''

t0 = time.perf_counter()
kernel = cp.RawKernel(
    cupy_source,
    "affine_transform",
    options=("--std=c++14",),
    backend="nvrtc",
)
cupy_input = cp.arange(N, dtype=cp.float64)
cupy_output = cp.empty_like(cupy_input)
kernel(
    (BLOCKS,),
    (THREADS,),
    (cupy_input, cupy_output, np.int64(N)),
)
cp.cuda.runtime.deviceSynchronize()
elapsed = time.perf_counter() - t0

cupy_expected = 2.0 * np.arange(N, dtype=np.float64) + 1.0
np.testing.assert_allclose(
    cp.asnumpy(cupy_output),
    cupy_expected,
    rtol=1e-13,
    atol=1e-13,
)

print(f"First compile and launch time: {elapsed:.6f} s")
print("First output values:", cp.asnumpy(cupy_output[:5]))
print("CuPy RawKernel NVRTC compilation: PASS")

print("\n" + "=" * 72)
print("NUMBA CUDA JIT VALIDATION")
print("=" * 72)
print("CUDA available:", cuda.is_available())
if not cuda.is_available():
    raise RuntimeError("Numba CUDA cannot access a GPU")

device = cuda.get_current_device()
print("Current device:", device.name)
print("Compute capability:", device.compute_capability)

@cuda.jit
def numba_affine_transform(input_array, output_array):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(start, input_array.size, stride):
        output_array[i] = 3.0 * input_array[i] - 2.0

host_input = np.arange(N, dtype=np.float64)
device_input = cuda.to_device(host_input)
device_output = cuda.device_array_like(device_input)

t0 = time.perf_counter()
numba_affine_transform[BLOCKS, THREADS](device_input, device_output)
cuda.synchronize()
elapsed = time.perf_counter() - t0

host_output = device_output.copy_to_host()
numba_expected = 3.0 * host_input - 2.0
np.testing.assert_allclose(
    host_output,
    numba_expected,
    rtol=1e-13,
    atol=1e-13,
)

print(f"First JIT and launch time: {elapsed:.6f} s")
print("First output values:", host_output[:5])
print("Compiled signatures:", numba_affine_transform.signatures)
print("Numba CUDA JIT compilation: PASS")

print("\n" + "=" * 72)
print("ALL GPU VALIDATION TESTS PASSED")
print("=" * 72)
print("MPI and parallel h5py:             PASS")
print("PyTorch CUDA execution:           PASS")
print("CuPy RawKernel NVRTC compilation: PASS")
print("Numba CUDA JIT compilation:       PASS")
PY

echo
echo "===== VALIDATION COMPLETED SUCCESSFULLY ====="
echo "Virtualenv: ${VENV_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "CuPy cache: ${CUPY_CACHE_DIR}"
