#!/bin/bash -l
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=boost_usr_prod
#SBATCH --time=0:30:00
#SBATCH --mem=0
#SBATCH --job-name=run_laplace_1node_prof
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:4
#SBATCH --qos=boost_qos_dbg
#SBATCH --exclusive

set -euo pipefail
mkdir -p logs profiles_${SLURM_JOB_ID} outputs_${SLURM_JOB_ID}

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

NSYS_COMMON="nsys profile \
  --force-overwrite=true \
  --stats=true \
  --sample=none \
  --cpuctxsw=none \
  --cuda-memory-usage=true"

${NSYS_COMMON} \
  --trace=cuda,nvtx,osrt \
  -o profiles_${SLURM_JOB_ID}/prof_laplace_cupy.job${SLURM_JOB_ID} \
  python laplace_cupy.py \
  --nx 40000 \
  --ny 40000 \
  --fixed-iters 100 \
  --output outputs_${SLURM_JOB_ID}/out_laplace_cupy_100.h5

${NSYS_COMMON} \
  --trace=cuda,nvtx,osrt \
  -o profiles_${SLURM_JOB_ID}/prof_laplace_cupy_raw.job${SLURM_JOB_ID} \
  python laplace_cupy_raw.py \
  --nx 40000 \
  --ny 40000 \
  --fixed-iters 100 \
  --tol -1.0 \
  --output outputs_${SLURM_JOB_ID}/out_laplace_cupy_raw_100.h5

srun ${NSYS_COMMON} \
  --trace=cuda,nvtx,osrt,mpi,ucx \
  --mpi-impl=openmpi \
  -o profiles_${SLURM_JOB_ID}/prof_laplace_cupy_nccl.rank%q{SLURM_PROCID}.job${SLURM_JOB_ID} \
  python laplace_cupy_nccl.py \
  --nx 40000 \
  --ny 40000 \
  --fixed-iters 100 \
  --tol -1.0 \
  --output outputs_${SLURM_JOB_ID}/out_laplace_cupy_nccl_100.h5

srun ${NSYS_COMMON} \
  --trace=cuda,nvtx,osrt,mpi,ucx \
  --mpi-impl=openmpi \
  -o profiles_${SLURM_JOB_ID}/prof_laplace_cupy_mpi.rank%q{SLURM_PROCID}.job${SLURM_JOB_ID} \
  python laplace_cupy_mpi.py \
  --nx 40000 \
  --ny 40000 \
  --fixed-iters 100 \
  --tol -1.0 \
  --output outputs_${SLURM_JOB_ID}/out_laplace_cupy_mpi_100.h5

${NSYS_COMMON} \
  --trace=cuda,nvtx,osrt \
  -o profiles_${SLURM_JOB_ID}/prof_laplace_numba.job${SLURM_JOB_ID} \
  python laplace_numba.py \
  --nx 40000 \
  --ny 40000 \
  --fixed-iters 100 \
  --tol -1.0 \
  --output outputs_${SLURM_JOB_ID}/out_laplace_numba_100.h5

srun ${NSYS_COMMON} \
  --trace=cuda,nvtx,osrt,mpi,ucx \
  --mpi-impl=openmpi \
  -o profiles_${SLURM_JOB_ID}/prof_laplace_numba_mpi.rank%q{SLURM_PROCID}.job${SLURM_JOB_ID} \
  python laplace_numba_mpi.py \
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

