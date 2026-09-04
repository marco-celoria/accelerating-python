#!/bin/bash -l
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=boost_usr_prod
#SBATCH --time=0:30:00
#SBATCH --mem=0
#SBATCH --job-name=run_compare
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:4
#SBATCH --qos=boost_qos_dbg
#SBATCH --exclusive

set -euo pipefail

module purge
module load cuda/12.2
module load gcc/12.2.0-binut2.41
module load cmake/3.27.9
module load python/3.11.7
module load openmpi/5.0.9--gcc--12.2.0-cuda-12.2
module load nccl/2.22.3-1--gcc--12.2.0-cuda-12.2-spack0.22
module load gdrcopy/2.4.1-cuda-12.2

export UCX_TLS=rc,cuda_copy,cuda_ipc,self
export UCX_MEMTYPE_CACHE=n
export OMPI_MCA_pml=ucx

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
cd ..
source pygpu_cuda122/bin/activate
cd -

ID_JOB_DIR=55301309

echo "Numpy vs Cupy + NCCL"
python compare_h5.py outputs_${ID_JOB_DIR}/out_laplace_numpy_100.h5 outputs_${ID_JOB_DIR}/out_laplace_cupy_nccl_100.h5
echo "Numpy vs Cupy + MPI"
python compare_h5.py outputs_${ID_JOB_DIR}/out_laplace_numpy_100.h5 outputs_${ID_JOB_DIR}/out_laplace_cupy_mpi_100.h5
echo "Numpy vs Numba + MPI"
python compare_h5.py outputs_${ID_JOB_DIR}/out_laplace_numpy_100.h5 outputs_${ID_JOB_DIR}/out_laplace_numba_mpi_100.h5
echo "Numpy vs Cupy"
python compare_h5.py outputs_${ID_JOB_DIR}/out_laplace_numpy_100.h5 outputs_${ID_JOB_DIR}/out_laplace_cupy_100.h5
echo "Numpy vs Raw Cupy"
python compare_h5.py outputs_${ID_JOB_DIR}/out_laplace_numpy_100.h5 outputs_${ID_JOB_DIR}/out_laplace_cupy_raw_100.h5
echo "Numpy vs Numba Cuda"
python compare_h5.py outputs_${ID_JOB_DIR}/out_laplace_numpy_100.h5 outputs_${ID_JOB_DIR}/out_laplace_numba_100.h5
