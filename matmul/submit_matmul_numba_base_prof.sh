#!/bin/bash -l
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --partition=boost_usr_prod
#SBATCH --time=0:30:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --job-name=run_matmul_numba_base_prof
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --qos=boost_qos_dbg
#SBATCH --exclusive

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

mkdir -p reports_numba_base_${SLURM_JOB_ID}

nsys profile --trace=cuda,cublas,nvtx,osrt  -o nsys_prof_matmul_numba_base.%p python matmul_numba_base.py 10000 10000 10000 10

ncu --set full --section SpeedOfLight_HierarchicalTensorRooflineChart -o ncu_prof_matmul_numba_base.%p python matmul_numba_base.py 10000 10000 10000 1

mv  ncu_prof_matmul_numba_base*.ncu-rep  reports_numba_base_${SLURM_JOB_ID}
mv nsys_prof_matmul_numba_base*.nsys-rep reports_numba_base_${SLURM_JOB_ID}

