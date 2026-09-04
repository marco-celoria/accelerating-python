#!/bin/bash -l
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --partition=boost_usr_prod
#SBATCH --time=0:30:00
#SBATCH --mem=0
#SBATCH --job-name=run_intro
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:4
#SBATCH --qos=boost_qos_dbg
#SBATCH --exclusive

set -euo pipefail
mkdir -p logs profiles_${SLURM_JOB_ID} 

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
  --sample=none \
  --cpuctxsw=none \
  --cuda-memory-usage=true"

python intro_cupy_numba_01.py
python intro_cupy_numba_02.py
python intro_cupy_numba_03.py
${NSYS_COMMON} --trace=cuda,nvtx,osrt -o profiles_${SLURM_JOB_ID}/nsys_prof_intro_04_pinned.job${SLURM_JOB_ID} python intro_cupy_numba_04.py
python intro_cupy_numba_05.py
${NSYS_COMMON} --trace=cuda,nvtx,osrt -o profiles_${SLURM_JOB_ID}/nsys_prof_intro_06_nonfused.job${SLURM_JOB_ID} python intro_cupy_numba_06.py
${NSYS_COMMON} --trace=cuda,nvtx,osrt -o profiles_${SLURM_JOB_ID}/nsys_prof_intro_07_fused.job${SLURM_JOB_ID} python intro_cupy_numba_07.py
python intro_cupy_numba_08.py
python intro_cupy_numba_09.py
${NSYS_COMMON} --trace=cuda,nvtx,osrt -o profiles_${SLURM_JOB_ID}/nsys_prof_intro_10_raw.job${SLURM_JOB_ID} python intro_cupy_numba_10.py
ncu --set full  -o profiles_${SLURM_JOB_ID}/ncu_prof_intro_11_tensorcores.job${SLURM_JOB_ID} python intro_cupy_numba_11.py
python intro_cupy_numba_12.py
${NSYS_COMMON} --trace=cuda,nvtx,osrt -o profiles_${SLURM_JOB_ID}/nsys_prof_intro_13_pinned.job${SLURM_JOB_ID} python intro_cupy_numba_13.py
python intro_cupy_numba_14.py
python intro_cupy_numba_15.py
python intro_cupy_numba_16.py
python intro_cupy_numba_17.py
python intro_cupy_numba_18.py

${NSYS_COMMON} --trace=cuda,nvtx,osrt -o profiles_${SLURM_JOB_ID}/nsys_prof_cupy_cooperative_normalize.job${SLURM_JOB_ID} python cupy_cooperative_normalize.py
ncu --set full  -o profiles_${SLURM_JOB_ID}/ncu_prof_cupy_cooperative_normalize.job${SLURM_JOB_ID} python cupy_cooperative_normalize.py

${NSYS_COMMON} --trace=cuda,nvtx,osrt -o profiles_${SLURM_JOB_ID}/nsys_prof_numba_cooperative_normalize.job${SLURM_JOB_ID} python numba_cooperative_normalize.py
ncu --set full  -o profiles_${SLURM_JOB_ID}/ncu_prof_numba_cooperative_normalize.job${SLURM_JOB_ID} python numba_cooperative_normalize.py

