#!/bin/bash -l
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --partition=boost_usr_prod
#SBATCH --time=0:30:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --job-name=run_matmul_cupy_shared_test
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

python matmul_cupy_shared.py 999 623 379 10
python matmul_cupy_shared.py 999 379 623 10
python matmul_cupy_shared.py 623 999 379 10
python matmul_cupy_shared.py 623 379 999 10
python matmul_cupy_shared.py 379 999 623 10
python matmul_cupy_shared.py 379 623 999 10
python matmul_cupy_shared.py 999 999 999 10
python matmul_cupy_shared.py 623 623 623 10
python matmul_cupy_shared.py 379 379 379 10

