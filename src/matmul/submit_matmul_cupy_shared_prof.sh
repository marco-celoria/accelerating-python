#!/bin/bash -l
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --partition=boost_usr_prod
#SBATCH --time=0:30:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --job-name=run_matmul_cupy_shared_prof
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

mkdir -p reports_cupy_shared_${SLURM_JOB_ID}

nsys profile --trace=cuda,cublas  -o nsys_prof_matmul_cupy_shared.%p python matmul_cupy_shared.py 10000 10000 10000 10

ncu --set full --section  SpeedOfLight_HierarchicalTensorRooflineChart  -o ncu_prof_matmul_cupy_shared.%p python matmul_cupy_shared.py 10000 10000 10000 1

mv  ncu_prof_matmul_cupy_shared*.ncu-rep  reports_cupy_shared_${SLURM_JOB_ID}
mv nsys_prof_matmul_cupy_shared*.nsys-rep reports_cupy_shared_${SLURM_JOB_ID}

