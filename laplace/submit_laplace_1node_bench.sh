#!/bin/bash -l
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=boost_usr_prod
#SBATCH --time=0:30:00
#SBATCH --mem=0
#SBATCH --job-name=run_laplace_1node_bench
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:4
#SBATCH --qos=boost_qos_dbg
#SBATCH --exclusive

set -euo pipefail
mkdir -p logs outputs_${SLURM_JOB_ID}

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

python laplace_cupy.py \
  --nx 40000 \
  --ny 40000 \
  --fixed-iters 100 \
  --output outputs_${SLURM_JOB_ID}/out_laplace_cupy_100.h5

 python laplace_cupy_raw.py \
  --nx 40000 \
  --ny 40000 \
  --fixed-iters 100 \
  --tol -1.0 \
  --output outputs_${SLURM_JOB_ID}/out_laplace_cupy_raw_100.h5

srun python laplace_cupy_mpi.py \
  --nx 40000 \
  --ny 40000 \
  --fixed-iters 100 \
  --tol -1.0 \
  --output outputs_${SLURM_JOB_ID}/out_laplace_cupy_mpi_100.h5

srun python laplace_cupy_nccl.py \
  --nx 40000 \
  --ny 40000 \
  --fixed-iters 100 \
  --tol -1.0 \
  --output outputs_${SLURM_JOB_ID}/out_laplace_cupy_nccl_100.h5

python laplace_numba.py \
  --nx 40000 \
  --ny 40000 \
  --fixed-iters 100 \
  --tol -1.0 \
  --output outputs_${SLURM_JOB_ID}/out_laplace_numba_100.h5

srun python laplace_numba_mpi.py \
  --nx 40000 \
  --ny 40000 \
  --fixed-iters 100 \
  --tol -1.0 \
  --output outputs_${SLURM_JOB_ID}/out_laplace_numba_mpi_100.h5

python laplace_numpy.py \
  --nx 40000 \
  --ny 40000 \
  --fixed-iters 100 \
  --tol -1.0 \
  --output outputs_${SLURM_JOB_ID}/out_laplace_numpy_100.h5

echo "Numpy vs Cupy + NCCL"
python compare_h5.py outputs_${SLURM_JOB_ID}/out_laplace_numpy_100.h5 outputs_${SLURM_JOB_ID}/out_laplace_cupy_nccl_100.h5
echo "Numpy vs Cupy + MPI"
python compare_h5.py outputs_${SLURM_JOB_ID}/out_laplace_numpy_100.h5 outputs_${SLURM_JOB_ID}/out_laplace_cupy_mpi_100.h5
echo "Numpy vs Numba + MPI"
python compare_h5.py outputs_${SLURM_JOB_ID}/out_laplace_numpy_100.h5 outputs_${SLURM_JOB_ID}/out_laplace_numba_mpi_100.h5
echo "Numpy vs Cupy"
python compare_h5.py outputs_${SLURM_JOB_ID}/out_laplace_numpy_100.h5 outputs_${SLURM_JOB_ID}/out_laplace_cupy_100.h5
echo "Numpy vs Raw Cupy"
python compare_h5.py outputs_${SLURM_JOB_ID}/out_laplace_numpy_100.h5 outputs_${SLURM_JOB_ID}/out_laplace_cupy_raw_100.h5
echo "Numpy vs Numba Cuda"
python compare_h5.py outputs_${SLURM_JOB_ID}/out_laplace_numpy_100.h5 outputs_${SLURM_JOB_ID}/out_laplace_numba_100.h5

