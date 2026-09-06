#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=boost_usr_prod
#SBATCH --time=0:30:00
#SBATCH --mem=0
#SBATCH --job-name=run_matmul_cupy_base_prof
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --qos=boost_qos_dbg
#SBATCH --exclusive

set -euo pipefail
mkdir -p logs profiles_cupy_base_${SLURM_JOB_ID}

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

# ---------------------------------------------------------------------------
# User configuration
# ---------------------------------------------------------------------------

cd ..
BASE_DIR="$PWD"
VENV_DIR="${BASE_DIR}/pygpu_cuda122"
cd -
# ---------------------------------------------------------------------------
# Select the GCC 12.2 runtime libraries
# ---------------------------------------------------------------------------

GCC_LIBSTDCPP="$(readlink -f "$(g++ -print-file-name=libstdc++.so.6)")"
GCC_LIBGCC="$(readlink -f "$(gcc -print-file-name=libgcc_s.so.1)")"
GCC_RUNTIME_DIR="$(dirname "${GCC_LIBSTDCPP}")"
export GCC_RUNTIME_DIR
export LD_PRELOAD="${GCC_LIBSTDCPP}:${GCC_LIBGCC}${LD_PRELOAD:+:${LD_PRELOAD}}"

# ---------------------------------------------------------------------------
# MPI, CUDA, HDF5, and application environment
# ---------------------------------------------------------------------------

CUDA_ROOT="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
export CUDA_HOME="${CUDA_ROOT}"
export CUDA_PATH="${CUDA_ROOT}"
export CUDACXX="${CUDA_ROOT}/bin/nvcc"

export HDF5_HOME="${BASE_DIR}/hdf5-1.14.3-openmpi5"
export HDF5_DIR="${HDF5_HOME}"

export PATH="${HDF5_HOME}/bin:${CUDA_ROOT}/bin:${PATH}"
export CPATH="${CUDA_ROOT}/include${CPATH:+:${CPATH}}"
export LIBRARY_PATH="${GCC_RUNTIME_DIR}:${CUDA_ROOT}/lib64:${CUDA_ROOT}/targets/x86_64-linux/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export LD_LIBRARY_PATH="${GCC_RUNTIME_DIR}:${HDF5_HOME}/lib:${HDF5_HOME}/lib64:${CUDA_ROOT}/lib64:${CUDA_ROOT}/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

export CUPY_ACCELERATORS="cub"
export CUPY_CACHE_DIR="${BASE_DIR}/.cupy/kernel_cache"

export MPI4PY_RC_THREAD_LEVEL=funneled
export UCX_TLS=rc,cuda_copy,cuda_ipc,self
export UCX_MEMTYPE_CACHE=n
export OMPI_MCA_pml=ucx

source "${VENV_DIR}/bin/activate"

nsys profile --trace=cuda,cublas,nvtx,osrt  -o nsys_matmul_cupy_base.%p python matmul_cupy_base.py 10000 10000 10000 10

ncu --set full --section SpeedOfLight_HierarchicalTensorRooflineChart -o ncu_matmul_cupy_base.%p python matmul_cupy_base.py 10000 10000 10000 1

mv  ncu_matmul_cupy_base*.ncu-rep  profiles_cupy_base_${SLURM_JOB_ID}
mv nsys_matmul_cupy_base*.nsys-rep profiles_cupy_base_${SLURM_JOB_ID}

