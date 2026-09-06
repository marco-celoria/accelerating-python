# Accelerating Python with GPUs: CuPy and Numba CUDA Examples

This project is a practical collection of CUDA Python examples, benchmarks, and scientific solvers developed for the **Leonardo Booster** environment. It progresses from introductory NumPy and CuPy operations to custom CUDA kernels, multi-stream execution, cross-library interoperability, cooperative kernels, CUDA Graphs, Tensor Core programming, matrix-multiplication benchmarks, and single-node or distributed Laplace solvers.

The examples accompany the guide **Accelerating Python: A Guide to GPU Optimization with CuPy and Numba**.

## Project structure

```text
.
|-- intro/                       Introductory NumPy, CuPy, and Numba CUDA examples
|-- advanced/                    Streams, asynchronous transfers, interoperability,
|                               cooperative kernels, CUDA Graphs, and FP64 WMMA
|-- matmul/                      Consistent matrix-multiplication benchmarks
|-- laplace/                     Serial and distributed 2D Laplace/Jacobi solvers
|-- src/                         Project source material
|-- sources/                     Cached third-party source archives and repositories
|-- wheels/                      Locally built Python wheels, including PyTorch
|-- pygpu_cuda122/               Unified CUDA 12.2 Python virtual environment
|-- hdf5-1.14.3-openmpi5/        User-built parallel HDF5 installation
|-- build-hdf5-1.14.3/           HDF5 build tree
|-- logs/                        Slurm installation and execution logs
|-- requirements.txt             Direct Python requirements
|-- requirements-lock.txt        Fully resolved Python environment
|-- installed_packages.txt       Installed-package inventory
|-- submit_install_1.sh          CPU installation stage
|-- submit_install_2.sh          GPU build and validation stage
|-- submit_run.sh                Main execution submission script
|-- submit_validate_gpu.sh       GPU environment validation
|-- reinstall_all.sh             Environment reinstallation helper
`-- reset_installation.sh        Installation cleanup/reset helper
```

Generated environments, build trees, caches, wheels, logs, and downloaded sources are deployment artifacts rather than core example code.

## Learning path

The recommended order is:

1. **`intro/`**: NumPy-to-CuPy migration, transfers, built-in operations, fusion, custom kernels, Numba indexing, grid-stride loops, reductions, timing, NVTX, and Nsight Systems.
2. **`advanced/`**: CUDA streams and events, pinned-memory pipelines, CuPy-Numba and CuPy-PyTorch interoperability, cooperative groups, CUDA Graph replay, and FP64 WMMA.
3. **`matmul/`**: comparable FP32 and FP64 GEMM benchmarks using naive and shared-memory kernels in Numba CUDA and CuPy RawModule, plus optimized CuPy `matmul`.
4. **`laplace/`**: consistent FP64 Jacobi solvers using NumPy, vectorized CuPy, Numba CUDA, CuPy RawKernel, CUDA-aware MPI, and NCCL communication with overlap.

Each directory contains its own detailed `README.md` with implementation notes, execution examples, profiling guidance, validation procedures, and resource considerations.

## Software environment

The unified environment is built for the Leonardo CUDA 12.2 software stack and includes:

- Python 3.11;
- CUDA 12.2;
- GCC 12.2;
- Open MPI 5.0.9 with CUDA support;
- NCCL 2.22.3;
- cuDNN 8.9.7;
- parallel HDF5 1.14.3;
- NumPy, SciPy, Matplotlib, CuPy, Numba, Numba CUDA, mpi4py, and MPI-enabled h5py;
- PyTorch 2.5.1 built locally for CUDA 12.2, A100 compute capability 8.0, system NCCL, MPI, Gloo, and cuDNN;
- NVIDIA Nsight Systems and Nsight Compute for profiling.

Exact Python package versions are recorded in `requirements-lock.txt`.

## Installation

Installation is split into two dependent Slurm jobs.

### Stage 1: CPU build and environment preparation

`submit_install_1.sh` runs on `lrd_all_serial` and:

- loads the Leonardo compiler, CUDA, MPI, NCCL, GDRCopy, and cuDNN modules;
- verifies the CUDA development toolkit and GCC runtime;
- locates the system NCCL and cuDNN installations;
- downloads and builds parallel HDF5 1.14.3 with Open MPI 5;
- creates the `pygpu_cuda122` virtual environment;
- installs the scientific, CUDA Python, CuPy, Numba, mpi4py, and MPI-enabled h5py stack;
- downloads and prepares the PyTorch 2.5.1 source tree.

### Stage 2: GPU PyTorch build and validation

`submit_install_2.sh` runs on one exclusive Leonardo Booster GPU and:

- validates the artifacts produced by stage 1;
- builds a CUDA-enabled PyTorch 2.5.1 wheel for NVIDIA A100 GPUs;
- installs the wheel from the local `wheels/` directory;
- verifies CUDA, cuDNN, NCCL, MPI, Gloo, CuPy, and parallel-HDF5 support;
- confirms that PyTorch dynamically resolves NCCL from the loaded Leonardo module;
- writes the resolved Python environment to `requirements-lock.txt`.

Submit both stages from the project root:

```bash
mkdir -p logs

job1=$(sbatch --parsable submit_install_1.sh)
echo "Submitted installation stage 1 as job ${job1}"

job2=$(sbatch --parsable \
    --dependency=afterok:"${job1}" \
    submit_install_2.sh)
echo "Submitted installation stage 2 as job ${job2}"
```

Stage 2 starts only if stage 1 completes successfully.

Monitor the jobs with:

```bash
squeue -j "${job1},${job2}"
```

Inspect their final states with:

```bash
sacct -j "${job1},${job2}" \
    --format=JobID,JobName,State,Elapsed,ExitCode
```

Installation logs are written under `logs/` using the Slurm job name and job ID.

## Activating the environment

After loading the same Leonardo modules used by the run scripts, activate the unified environment from the project root:

```bash
source pygpu_cuda122/bin/activate
```

The provided submission scripts configure the complete runtime environment, including CUDA paths, the GCC runtime, parallel HDF5, CuPy's persistent kernel cache, CUB acceleration, and CUDA-aware UCX transports.

GPU programs should be executed on allocated compute nodes, not directly on a login node.

## Running the examples

Consult the README in the relevant directory, then run an individual program or submit its local Slurm script.

Examples:

```bash
# Introductory example
cd intro
python intro_cupy_rawkernel.py

# Advanced example
cd ../advanced
python advanced_cupy_graph.py

# Matrix-multiplication benchmark
cd ../matmul
python matmul_cupy_shared.py 2048 2048 2048 20 \
    --dtype float32 \
    --tpb 16

# Single-GPU Laplace solver
cd ../laplace
python laplace_cupy_raw.py \
    --nx 2048 \
    --ny 2048 \
    --fixed-iters 1000 \
    --check-every 100 \
    --device 0
```

Use the project submission and validation helpers when running the complete workflow:

```bash
sbatch submit_validate_gpu.sh
sbatch submit_run.sh
```

Review each script's Slurm resource requests and site-specific assumptions before submission.

## Profiling

The examples use NVTX annotations and are designed for inspection with:

- **Nsight Systems** for CUDA timelines, streams, event dependencies, memory transfers, synchronization, MPI/NCCL behavior, and launch overhead;
- **Nsight Compute** for kernel-level instruction, memory, occupancy, scheduler, and Tensor Core metrics.

Warm-up generally occurs before the principal timed or profiled region so that JIT compilation and first-use initialization can be distinguished from steady-state execution. Profiling adds overhead and should be kept separate from uninstrumented performance measurements.

## Reproducibility and validation

The project emphasizes:

- explicit warm-up before timing JIT-compiled operations;
- CUDA events for GPU timing and wall clocks for host-observed execution;
- preallocated outputs where practical;
- seeded inputs in benchmark programs;
- full-array numerical checks with dtype-appropriate tolerances;
- fixed-iteration modes for comparable solver runs;
- HDF5 output with geometry, backend, convergence, and decomposition metadata;
- validation across NumPy, CuPy, Numba CUDA, MPI, and NCCL implementations;
- persistent package and kernel caches;
- a frozen Python environment in `requirements-lock.txt`.

Floating-point reductions can differ slightly across implementations because operation and summation orders are not identical. Use combined relative and absolute tolerances rather than exact equality.

## Scope

The repository is intended for education, implementation comparison, profiling exercises, and HPC experimentation. The custom kernels favor clarity and explicit CUDA concepts over exhaustive optimization. Optimized library operations such as CuPy matrix multiplication remain the appropriate performance reference where applicable.

For detailed commands and design notes, see:

- [`intro/README.md`](intro/README.md)
- [`advanced/README.md`](advanced/README.md)
- [`matmul/README.md`](matmul/README.md)
- [`laplace/README.md`](laplace/README.md)
