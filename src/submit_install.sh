#!/bin/bash -l
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --partition=lrd_all_serial
#SBATCH --time=4:00:00
#SBATCH --mem=30GB
#SBATCH --job-name=job_install
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

mkdir -p logs

BASE_DIR=$(pwd)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}


# -----------------------------
# USER CONFIG (edit if needed)
# -----------------------------
HDF5_VERSION=1.14.3
INSTALL_PREFIX=$BASE_DIR/hdf5-${HDF5_VERSION}-openmpi5
BUILD_DIR=$BASE_DIR/build-hdf5-${HDF5_VERSION}

module purge
module load cuda/12.2
module load gcc/12.2.0-binut2.41
module load cmake/3.27.9
module load python/3.11.7
module load openmpi/5.0.9--gcc--12.2.0-cuda-12.2
module load cutensor/1.5.0.3--gcc--12.2.0
module load cudnn/8.9.7.29-12--gcc--12.2.0-cuda-12.2
module load nccl/2.22.3-1--gcc--12.2.0-cuda-12.2-spack0.22
module load gdrcopy/2.4.1-cuda-12.2




# -----------------------------
# PREPARE BUILD
# -----------------------------
echo "Preparing directories..."

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# -----------------------------
# DOWNLOAD SOURCE
# -----------------------------
echo "Downloading HDF5 ${HDF5_VERSION}..."

if [ ! -f hdf5.tar.gz ]; then
    wget -O hdf5.tar.gz https://github.com/HDFGroup/hdf5/archive/refs/tags/hdf5-1_14_3.tar.gz
fi

tar xf hdf5.tar.gz
cd hdf5-hdf5-1_14_3

# -----------------------------
# CONFIGURE
# -----------------------------
echo "Configuring HDF5..."

mkdir -p build
cd build

cmake .. \
  -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
  -DCMAKE_BUILD_TYPE=Release \
  -DHDF5_ENABLE_PARALLEL=ON \
  -DHDF5_BUILD_FORTRAN=ON \
  -DHDF5_BUILD_HL_LIB=ON \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_C_COMPILER=mpicc \
  -DCMAKE_CXX_COMPILER=mpicxx \
  -DCMAKE_Fortran_COMPILER=mpifort

# -----------------------------
# BUILD
# -----------------------------
echo "Building..."

make -j${SLURM_CPUS_PER_TASK}

# -----------------------------
# INSTALL
# -----------------------------
echo "Installing to ${INSTALL_PREFIX}..."

make install

# -----------------------------
# VERIFY
# -----------------------------
echo "Verifying installation..."

${INSTALL_PREFIX}/bin/h5pcc -showconfig | grep -i parallel || true

echo ""
echo "===== DONE ====="
echo "HDF5 installed in: ${INSTALL_PREFIX}"
echo ""

export HDF5_HOME=$INSTALL_PREFIX
export PATH=${HDF5_HOME}/bin:$PATH
export LD_LIBRARY_PATH=${HDF5_HOME}/lib:$LD_LIBRARY_PATH

cd $BASE_DIR

export MAKEFLAGS="-j${SLURM_CPUS_PER_TASK}"

python -m venv pygpu_venv --system-site-packages
source pygpu_venv/bin/activate
python -m pip install -U setuptools pip
pip install -vvv --no-cache-dir -r requirements.txt
HDF5_MPI="ON" CC=mpicc HDF5_DIR=${HDF5_HOME} pip install -vvv --no-cache-dir --no-binary=h5py h5py
python -m cupyx.tools.install_library --cuda 12.x --library cutensor --prefix "${BASE_DIR}/pygpu_venv/.cupy/cuda_lib"
python -m cupyx.tools.install_library --cuda 12.x --library nccl     --prefix "${BASE_DIR}/pygpu_venv/.cupy/cuda_lib"


