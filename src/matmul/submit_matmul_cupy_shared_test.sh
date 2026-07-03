#!/bin/bash -l
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --partition=boost_usr_prod
#SBATCH --time=0:30:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --job-name=run_matmul_cupy_shared_test
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --qos=boost_qos_dbg
#SBATCH --exclusive

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

module purge
module load cuda/12.2
module load gcc/12.2.0-binut2.41
module load cmake/3.27.9
module load hdf5/1.14.3--gcc--12.2.0-spack0.22
module load python/3.11.7
module load openmpi/4.1.6--gcc--12.2.0
module load cutensor/1.5.0.3--gcc--12.2.0
module load cudnn/8.9.7.29-12--gcc--12.2.0-cuda-12.2
cd ..
source pygpu_venv/bin/activate
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

