#!/bin/bash -l
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=boost_usr_prod
#SBATCH --time=0:30:00
#SBATCH --mem=0
#SBATCH --job-name=run_info
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

ID_JOB_DIR=54792735

echo "----------------------------------------------------------------------------------------"
python hdf5_version.py
echo "----------------------------------------------------------------------------------------"
nvidia-smi > logs/nvidia_smi.txt
python -m pip freeze > logs/pip_free.txt
module list 2> logs/modules_used.txt
nvcc --version > logs/nvcc_version.txt
mpirun --version > logs/mpi_version.txt
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}" > logs/threads_info.txt
ucx_info -v > logs/ucx_info.txt
nvidia-smi --query-gpu=gpu_name,mig.mode.current --format=csv > logs/mig.txt

python numba_cuda_version.py
