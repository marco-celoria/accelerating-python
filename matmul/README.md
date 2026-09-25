# CUDA Matrix Multiplication Benchmarks with CuPy and Numba

This directory contains five GPU matrix-multiplication benchmarks implemented with CuPy and Numba CUDA. The programs compare naive global-memory kernels, shared-memory tiled kernels, and CuPy's optimized built-in matrix multiplication.

All implementations use the same matrix convention:

```text
A: (n, l)
B: (l, m)
C: (n, m)

C = A @ B
```

The benchmarks support rectangular matrices, `float32` and `float64`, reproducible random inputs, warm-up execution, compute-only timing, correctness verification against NumPy, and optional NVTX annotations.

## Contents

- [`matmul_numba_base.py`](#matmul_numba_basepy): naive Numba CUDA kernel using global memory
- [`matmul_numba_shared.py`](#matmul_numba_sharedpy): tiled Numba CUDA kernel using shared memory
- [`matmul_cupy_base.py`](#matmul_cupy_basepy): naive CUDA C++ kernel compiled through `cupy.RawModule`
- [`matmul_cupy_shared.py`](#matmul_cupy_sharedpy): templated shared-memory CUDA C++ kernel compiled through `cupy.RawModule`
- [`matmul_cupy_builtin.py`](#matmul_cupy_builtinpy): CuPy built-in matrix multiplication, typically backed by an optimized CUDA library

## Benchmark overview

### `matmul_numba_base.py`

Implements a straightforward one-thread-per-output-element matrix multiplication with Numba CUDA.

Each CUDA thread:

1. identifies one element `C[row, col]`;
2. reads the corresponding row of `A` and column of `B` from global memory;
3. computes the complete dot product;
4. writes one value to `C`.

The block dimensions are configurable independently through `--tpb-x` and `--tpb-y`. No shared-memory tiling is used.

### `matmul_numba_shared.py`

Implements square shared-memory tiling with Numba CUDA. Each thread block cooperatively loads one tile of `A` and one tile of `B`, computes partial dot products from shared memory, and iterates over the complete inner dimension.

The tile dimension is the compile-time constant:

```python
TPB = 16
```

The kernel must be launched with exactly `(TPB, TPB)` threads per block. To test another tile size, edit `TPB` in the source before running the program. The implementation checks the resulting thread-block dimensions and shared-memory requirement against the active device limits.

### `matmul_cupy_base.py`

Implements the same naive one-thread-per-output-element algorithm as a templated CUDA C++ kernel. CuPy compiles the source at runtime through `cupy.RawModule` and retrieves separate `float` and `double` specializations.

The square CUDA block dimension is configurable with `--tpb`.

### `matmul_cupy_shared.py`

Implements a templated CUDA C++ tiled kernel with two statically sized shared-memory arrays:

```text
Atile[TILE][TILE]
Btile[TILE][TILE]
```

The selected `--tpb` value is used as both the CUDA block dimension and the compile-time template parameter `TILE`. CuPy therefore compiles specializations for the requested tile size at runtime. The program validates both the total number of threads per block and the shared-memory requirement.

### `matmul_cupy_builtin.py`

Uses:

```python
cp.matmul(a_d, b_d, out=c_d)
```

This is the optimized library baseline for the suite. Unlike the educational custom kernels, the built-in implementation may select highly tuned, architecture-dependent algorithms and GPU math modes.

## Requirements

A working NVIDIA CUDA environment is required, including:

- an NVIDIA GPU supported by the installed CUDA software stack;
- a compatible NVIDIA driver;
- Python 3;
- NumPy;
- Numba CUDA for the Numba programs;
- CuPy for the CuPy programs and optional NVTX ranges.

Install the Python packages using versions appropriate for the CUDA toolkit available on the target system. For example, in an existing CUDA-enabled Python environment:

```bash
python -m pip install numpy numba
```

Install the CuPy package matching the CUDA environment according to the deployment policy of the local system or HPC center.

Confirm that the active environment can access a GPU before running the benchmarks:

```bash
nvidia-smi
python -c "from numba import cuda; print(cuda.is_available())"
python -c "import cupy as cp; print(cp.cuda.runtime.getDeviceCount())"
```

On a managed HPC system, load the required CUDA and Python modules and activate the intended virtual environment before execution.

## Command-line interface

All programs use the same positional arguments:

```text
python PROGRAM.py [n] [m] [l] [iters] [options]
```

The positional argument order is important:

- `n`: rows of `A` and `C`;
- `m`: columns of `B` and `C`;
- `l`: columns of `A` and rows of `B`, which is the inner dimension;
- `iters`: number of timed benchmark iterations.

The default values are:

```text
n = 1000
m = 1000
l = 1000
iters = 10
dtype = float64
seed = 12345
```

Display the complete options for any program with:

```bash
python PROGRAM.py --help
```

## Quick start

Run every implementation with its default configuration:

```bash
python matmul_numba_base.py
python matmul_numba_shared.py
python matmul_cupy_base.py
python matmul_cupy_shared.py
python matmul_cupy_builtin.py
```

Run a square `2048 x 2048` FP32 test for 20 iterations:

```bash
python matmul_numba_base.py 2048 2048 2048 20 --dtype float32
python matmul_numba_shared.py 2048 2048 2048 20 --dtype float32
python matmul_cupy_base.py 2048 2048 2048 20 --dtype float32
python matmul_cupy_shared.py 2048 2048 2048 20 --dtype float32
python matmul_cupy_builtin.py 2048 2048 2048 20 --dtype float32
```

Run a rectangular multiplication:

```bash
python matmul_cupy_shared.py 1536 768 1024 20 --dtype float64 --tpb 16
```

This computes:

```text
A: (1536, 1024)
B: (1024, 768)
C: (1536, 768)
```

## Implementation-specific options

### Numba baseline block geometry

`matmul_numba_base.py` permits independent x and y block dimensions:

```bash
python matmul_numba_base.py 2048 2048 2048 20 \
    --dtype float32 \
    --tpb-x 32 \
    --tpb-y 8
```

The program rejects configurations that exceed the active GPU's maximum threads per block or block-dimension limits.

### Numba tiled kernel

`matmul_numba_shared.py` has no tile-size command-line option. Its tile and block dimensions are controlled by the source-level constant:

```python
TPB = 16
```

After changing `TPB`, rerun the program so Numba compiles the corresponding specialization.

### CuPy baseline block geometry

`matmul_cupy_base.py` uses a square block:

```bash
python matmul_cupy_base.py 2048 2048 2048 20 \
    --dtype float32 \
    --tpb 16
```

The block contains `tpb * tpb` threads.

### CuPy tiled kernel

For `matmul_cupy_shared.py`, `--tpb` controls both the square block dimension and the compile-time CUDA C++ tile size:

```bash
python matmul_cupy_shared.py 2048 2048 2048 20 \
    --dtype float32 \
    --tpb 16
```

Changing `--tpb` causes CuPy to compile a matching template specialization. Common experimental values are `8`, `16`, and `32`, subject to device thread-block and shared-memory limits. A larger tile is not automatically faster.

## Precision

Select the arithmetic precision with:

```bash
--dtype float32
```

or:

```bash
--dtype float64
```

The default is `float64`.

FP64 throughput depends strongly on GPU architecture and product class. Consumer and visualization GPUs can provide substantially lower FP64 throughput than datacenter HPC accelerators. Compare FP32 and FP64 results only with this hardware dependence in mind.

The validation tolerances are:

```text
float32: rtol=1.0e-4,  atol=1.0e-4
float64: rtol=1.0e-10, atol=1.0e-10
```

These are pragmatic benchmark tolerances rather than formal forward-error bounds. Numerical differences can depend on the inner dimension, input distribution, accumulation order, compiler behavior, GPU architecture, and selected math mode.

## Timing methodology

Each program performs a GPU warm-up before collecting timings:

- Numba programs compile the required kernel specialization during warm-up;
- CuPy RawModule programs compile and load the CUDA C++ code before timed iterations;
- the built-in CuPy program performs an initial `cp.matmul` call;
- every program warms up the NumPy reference multiplication.

Consequently, the reported iteration timings exclude JIT compilation, RawModule compilation, first-launch initialization, and the initial NumPy call.

### GPU timing

GPU execution is measured with CUDA events. The reported per-iteration GPU time covers the matrix-multiplication operation itself and excludes:

- host-to-device input transfers;
- device-to-host result transfers;
- input generation;
- device allocation and initialization;
- kernel compilation;
- correctness verification.

The programs also report a total event interval around the GPU benchmark loop and the sum of the individually measured iterations. These quantities need not be bit-for-bit identical because event recording and per-iteration synchronization introduce additional sequencing overhead.

### CPU timing

The NumPy reference is measured with `time.perf_counter()`. Its output array is preallocated and passed through `out=` so allocation is not part of the timed multiplication.

NumPy may use an optimized and multithreaded BLAS library. CPU timings and the reported NumPy/GPU compute-time ratio therefore depend on:

- CPU model;
- BLAS implementation;
- BLAS thread count;
- process and thread affinity;
- system load;
- matrix sizes and precision.

For reproducible CPU comparisons, record and control the relevant BLAS threading variables, such as those used by OpenBLAS, Intel MKL, BLIS, or the system-provided numerical libraries.

## Reported performance

The programs use the conventional dense GEMM operation count:

```text
FLOPs = 2 * n * l * m
```

Each multiplication and addition is counted as two floating-point operations. Median throughput is computed as:

```text
GFLOP/s = (2 * n * l * m * 1e-9) / median_time_seconds
```

The final report includes:

- matrix shapes and dtype;
- iteration count;
- mean, median, minimum, and sample standard deviation;
- conventional GEMM FLOP count;
- median GPU and CPU GFLOP/s;
- NumPy/GPU compute-time ratio;
- correctness status.

The median is used for the main throughput and ratio calculations because it is less sensitive to isolated timing outliers than the mean.

## Correctness validation

After benchmarking, each program copies the final GPU result to the host and compares it with NumPy's result for the same input arrays and dtype.

Validation reports:

- whether the GPU result is finite;
- whether the CPU result is finite;
- the L2 norm of the difference;
- the maximum absolute error;
- the relative L2 error;
- the selected `rtol` and `atol`;
- the result of `numpy.allclose`.

A program exits with status `0` when validation passes and status `1` when it fails. This makes the programs suitable for batch jobs and automated test workflows:

```bash
python matmul_cupy_shared.py 1024 768 513 10 --dtype float32
status=$?
printf 'exit status: %d\n' "$status"
```

Non-multiple dimensions are intentionally supported. The custom kernels use bounds checks, and the tiled kernels zero-fill out-of-range tile elements.

## Fair-comparison guidance

For meaningful comparisons:

1. Use identical `n`, `m`, `l`, `iters`, `dtype`, and `seed` values.
2. Run all programs on the same GPU and software environment.
3. Avoid comparing results collected under different power, clock, or allocation conditions.
4. Use sufficiently large matrices so event resolution and launch overhead do not dominate.
5. Use several iterations and focus primarily on the median.
6. Run a small rectangular case first to verify bounds handling.
7. Treat the built-in CuPy result as the optimized library reference, not as an algorithmically equivalent custom kernel.
8. Record BLAS thread settings when interpreting the CPU comparison.
9. Separate compute-only results from end-to-end measurements that include data transfers.

A useful functional test set is:

```bash
# Small square test
python matmul_numba_base.py 127 127 127 3 --dtype float32

# Rectangular test with dimensions not divisible by 16
python matmul_numba_shared.py 257 193 129 3 --dtype float64

# Configurable CuPy tile test
python matmul_cupy_shared.py 513 385 257 5 --dtype float32 --tpb 16

# Optimized reference
python matmul_cupy_builtin.py 513 385 257 5 --dtype float32
```

## NVTX profiling

The programs annotate major phases with NVTX ranges when CuPy NVTX support is available. If NVTX support cannot be loaded, the helper silently becomes a no-op.

Annotated regions include operations such as:

- GPU allocation and initialization;
- host-to-device and device-to-host copies;
- warm-up execution;
- GPU kernel or `matmul` launches;
- NumPy reference multiplication.

Capture a system timeline with Nsight Systems, for example:

```bash
nsys profile \
    --trace=cuda,nvtx,osrt \
    --stats=true \
    --output=matmul_cupy_shared \
    python matmul_cupy_shared.py 2048 2048 2048 20 \
        --dtype float32 \
        --tpb 16
```

Profile a custom kernel with Nsight Compute, for example:

```bash
ncu \
    --set full \
    --target-processes all \
    --kernel-name-base demangled \
    --output matmul_numba_shared \
    python matmul_numba_shared.py 2048 2048 2048 5 \
        --dtype float32
```

Full metric sets can be expensive and may replay kernels multiple times. Use profiling runs separately from benchmark runs used for final timing comparisons.

## Running under Slurm

The following is a generic single-GPU batch script. Adapt account, partition, module names, time limit, and environment activation to the target cluster:

```bash
#!/bin/bash
#SBATCH --job-name=matmul_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=matmul_%j.out
#SBATCH --error=matmul_%j.err

set -euo pipefail

module purge
module load cuda

source /path/to/venv/bin/activate

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

nvidia-smi
python --version

srun python matmul_cupy_builtin.py 4096 4096 4096 20 --dtype float32
```

On systems that allocate GPUs through a cluster-specific option rather than `--gres=gpu:1`, use the local scheduler syntax.

## Memory requirements

Ignoring temporary storage internal to libraries, the main matrices require approximately:

```text
bytes = itemsize * (n*l + l*m + n*m)
```

The host side also retains input arrays and a NumPy reference output. Peak memory usage is therefore higher than the three device matrices alone.

Before launching very large cases, estimate both host and device memory requirements. For example, `float64` uses twice as much matrix storage as `float32`.

## Interpretation of results

The custom kernels are educational benchmarks rather than replacements for optimized GEMM libraries.

The naive kernels repeatedly load matrix elements from global memory and do not exploit data reuse through shared-memory tiling. The tiled kernels reduce redundant global-memory traffic by reusing tiles within each thread block, but they remain deliberately simple. They do not implement the many optimizations commonly found in production GEMM libraries, such as:

- register blocking;
- vectorized loads;
- asynchronous copies;
- double-buffered pipelines;
- warp-specialized execution;
- architecture-specific instruction selection;
- Tensor Core kernels;
- extensive shape-dependent autotuning.

For this reason, `matmul_cupy_builtin.py` should generally provide the strongest GPU baseline. The value of the custom kernels is to expose CUDA indexing, tiling, synchronization, shared-memory reuse, runtime compilation, validation, and profiling in compact Python-driven examples.

## Reproducibility checklist

When publishing or comparing benchmark results, record:

- GPU model and count;
- NVIDIA driver version;
- CUDA runtime and toolkit versions;
- Python version;
- CuPy, NumPy, Numba, and Numba CUDA versions;
- matrix dimensions and dtype;
- block or tile dimensions;
- iteration count and random seed;
- CPU model and BLAS implementation;
- BLAS and OpenMP thread settings;
- Slurm resource allocation and affinity settings;
- profiler usage, if any;
- GPU clock, power, and persistence settings when controlled.

Useful environment commands include:

```bash
nvidia-smi
python --version
python -c "import numpy; print('NumPy', numpy.__version__)"
python -c "import numba; print('Numba', numba.__version__)"
python -c "import cupy; print('CuPy', cupy.__version__); cupy.show_config()"
```

## Known limitations

- The benchmarks measure compute only, not end-to-end application time.
- A synchronization is performed for every measured GPU iteration, which provides simple independent samples but adds host-side overhead outside the event interval.
- The NumPy/GPU ratio is not a hardware-only speedup metric because the CPU reference depends on BLAS configuration.
- The naive kernels intentionally lack shared-memory and register-level optimization.
- The Numba tiled implementation uses a source-level tile constant rather than a command-line tile parameter.
- Very small workloads may produce unstable timings or non-positive median times at the available event resolution.
- FP64 performance is highly hardware-dependent.
- Validation tolerances are practical defaults and may need reconsideration for extremely large inner dimensions or differently scaled inputs.

## Exit behavior

Each program:

- returns `0` when the numerical validation passes;
- returns `1` when validation fails;
- raises an exception for invalid dimensions, unsupported launch geometry, insufficient shared memory, unavailable CUDA support, or non-positive timing results.

This behavior allows failures to propagate naturally in Slurm jobs, shell scripts, and continuous-integration workflows.
