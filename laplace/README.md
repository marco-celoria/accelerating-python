# 2D Laplace Solver Benchmarks with NumPy, CuPy, and Numba CUDA

This directory contains seven consistent implementations of a two-dimensional Laplace equation solver based on Jacobi iteration, plus an HDF5 comparison utility:

- a serial vectorized NumPy reference;
- a vectorized CuPy GPU implementation;
- a custom Numba CUDA implementation;
- a custom CuPy `RawKernel` implementation;
- a distributed Numba CUDA implementation with CUDA-aware MPI;
- a distributed CuPy `RawKernel` implementation with CUDA-aware MPI;
- a distributed CuPy `RawKernel` implementation with overlapped NCCL communication.

The programs use the same global grid layout, anisotropic finite-difference discretization, Dirichlet boundary conditions, ping-pong iteration, convergence metric, warm-up procedure, throughput definition, and optional HDF5 output. They are intended for correctness comparisons, GPU optimization studies, profiling, and reproducible fixed-iteration benchmarks.

## Contents

- `laplace_numpy.py`: serial vectorized NumPy reference
- `laplace_cupy.py`: single-GPU vectorized CuPy solver
- `laplace_numba.py`: single-GPU Numba CUDA solver with custom Jacobi and reduction kernels
- `laplace_cupy_raw.py`: single-GPU CuPy solver with custom CUDA `RawKernel` Jacobi and partial-reduction kernels
- `laplace_numba_mpi.py`: distributed Numba CUDA solver with CUDA-aware MPI halos
- `laplace_cupy_mpi.py`: distributed CuPy RawKernel solver with CUDA-aware MPI halos and global reduction
- `laplace_cupy_nccl.py`: distributed CuPy RawKernel solver with overlapped NCCL halos and NCCL all-reduce
- `compare_h5.py`: validates a candidate HDF5 solution against a reference

## Problem definition

The programs solve the homogeneous two-dimensional Laplace equation on a rectangular domain:

```text
nabla^2 u = 0
```

with physical dimensions:

```text
0 <= x <= lx
0 <= y <= ly
```

The global solution array has shape:

```text
(ny, nx)
```

where `nx` and `ny` include the physical boundary points. The grid spacing is:

```text
dx = lx / (nx - 1)
dy = ly / (ny - 1)
```

The implementations always use double precision:

```text
float64
```

The minimum supported grid is `3 x 3`, which contains one physical interior point.

## Array layout and boundaries

The row-major global array is interpreted as:

```text
u[0, :]   = bottom boundary
u[-1, :]  = top boundary
u[:, 0]   = left boundary
u[:, -1]  = right boundary
```

The boundary values are controlled by:

```text
--bc-left
--bc-right
--bc-bottom
--bc-top
```

The defaults are:

```text
left   = 0.0
right  = 1.0
bottom = 0.0
top    = 0.0
```

### Corner convention

Left and right boundaries are assigned first. Bottom and top boundaries are assigned afterwards. Therefore, bottom and top values take precedence at the four corners.

In assignment order:

```python
u[:, 0] = bc_left
u[:, -1] = bc_right
u[0, :] = bc_bottom
u[-1, :] = bc_top
```

All four implementations follow this convention. It is important to preserve the same ordering when comparing these programs with another serial, GPU, or distributed implementation.

## Jacobi discretization

For every physical interior point, the solver evaluates:

```text
u_new[i,j] = (
    (u[i,j-1] + u[i,j+1]) * dy^2
  + (u[i-1,j] + u[i+1,j]) * dx^2
) / (2 * (dx^2 + dy^2))
```

This form supports rectangular domains and unequal grid spacing. It reduces to the familiar average of the four nearest neighbors when `dx == dy`.

The implementations maintain two solution fields:

```text
u
unew
```

Only the physical interior of `unew` is updated. Both arrays are initialized with identical boundary values, so the Dirichlet boundaries remain valid when the references are exchanged after each iteration:

```python
u, unew = unew, u
```

The method is Jacobi iteration, not in-place Gauss-Seidel iteration.

## Convergence metric

The monitored convergence indicator is the relative L2 norm of the iterate update over the physical interior:

```text
||u_new - u||_2
-----------------
    ||u_new||_2
```

The denominator is protected against division by zero using a floor of `1.0e-300`.

This quantity is an **iterate-update norm**. It is not the discrete Laplace residual and should not be described as a residual norm. A small update norm indicates that successive Jacobi iterates are changing slowly, but its numerical value is not interchangeable with a residual-based stopping criterion.

The norm is evaluated:

- after iteration 1;
- every `--check-every` iterations;
- at the final permitted iteration.

Because tolerance-based termination can occur only when the norm is evaluated, `--check-every` affects both runtime and stopping granularity.

## Implementations

### NumPy reference

`laplace_numpy.py` uses vectorized NumPy expressions for the Jacobi stencil and reductions.

The implementation serves as the serial reference for:

- boundary-condition semantics;
- anisotropic discretization;
- iteration ordering;
- convergence evaluation;
- HDF5 output comparison.

The elapsed time is measured with `time.perf_counter()` and includes the monitored solver loop. NumPy expression temporaries may increase memory traffic and peak host memory beyond the two persistent solution fields.

### Vectorized CuPy

`laplace_cupy.py` translates the NumPy array expressions directly to CuPy. It provides a concise GPU implementation, but a single Jacobi expression can involve multiple generated kernels, temporary arrays, and multiple device-memory passes.

The convergence calculation uses CuPy reductions. The resulting scalar is copied to the host whenever the norm is checked because Python requires a host value for tolerance-based control flow. This scalar transfer synchronizes preceding work in the current CUDA stream.

The solver reports both:

- CUDA-event elapsed time;
- host wall-clock elapsed time.

### Numba CUDA

`laplace_numba.py` uses a custom Numba CUDA kernel for the Jacobi update. Its two-dimensional launch configuration is fixed in the source:

```python
TPB_X = 32
TPB_Y = 8
```

This produces a block of 256 threads. The x direction spans the contiguous row-major dimension, so a full warp maps across adjacent columns.

The update-norm calculation has two stages:

1. a custom grid-stride CUDA kernel computes per-block partial sums for the squared update and squared new iterate;
2. `cuda.reduce` completes the reductions over the compact partial arrays.

Reduction settings are:

```python
REDUCE_TPB = 256
REDUCE_BLOCKS_PER_SM = 4
```

The number of reduction blocks is limited to at most four blocks per streaming multiprocessor and never exceeds the number required to cover the interior.

### CuPy RawKernel

`laplace_cupy_raw.py` uses custom CUDA C++ kernels compiled and launched through CuPy:

- `jacobi_2d` performs the Jacobi stencil update;
- `rel_update_norm_partial_2d` computes per-block partial sums in dynamically allocated shared memory;
- `cp.sum` completes the two reductions.

Its launch constants match the Numba implementation:

```python
TPB_X = 32
TPB_Y = 8
REDUCE_TPB = 256
REDUCE_BLOCKS_PER_SM = 4
```

The custom kernels assume contiguous row-major arrays. Do not replace the persistent arrays with transposed or strided views unless the kernels are modified to account for strides.

The RawKernel implementation currently supports only `float64`.

## Distributed multi-GPU implementations

The three distributed programs use the same global grid, FP64 discretization, boundary conditions, ping-pong update, convergence metric, output layout, and throughput definition as the single-process solvers. They decompose the global grid into contiguous row slabs along the y direction. Each rank stores two halo rows:

```text
u[0, :]                    lower halo
u[1:local_ny + 1, :]       owned physical rows
u[local_ny + 1, :]         upper halo
```

The remainder-aware decomposition assigns one extra row to the first `ny % nranks` ranks. Bottom and top boundary assignments retain priority at the corners. Ranks that own only physical boundary rows still participate in communication even though they perform no Jacobi updates.

All distributed implementations use `MPI_THREAD_FUNNELED`, map one process to one GPU using launcher-local rank variables or an MPI shared-memory communicator, warm up computation and communication before timing, and reduce rank-local timing with `MPI.MAX`.

### `laplace_numba_mpi.py`

This version uses Numba CUDA for the Jacobi update and local norm reduction, CUDA-aware MPI `Sendrecv` for halo rows, and a host-buffer MPI `Allreduce` for the final two FP64 norm components.

The Jacobi kernel uses a `32 x 8` block. The norm path uses a 256-thread shared-memory reduction with a grid-stride loop, writes one pair of partial sums per block, and completes each local sum with `@cuda.reduce`. The reduction grid is capped at four blocks per SM to limit workspace and second-stage reduction work.

```text
CUDA-aware MPI halo exchange
             |
             v
Numba Jacobi kernel
             |
             v
shared-memory partial reduction
             |
             v
cuda.reduce on compact partial arrays
             |
             v
host MPI_Allreduce of two FP64 values
```

The default CUDA stream is synchronized before MPI accesses Numba device rows. Device fields and reduction workspaces remain allocated for the solver lifetime.

### `laplace_cupy_mpi.py`

This version uses CuPy `RawKernel` kernels for Jacobi and local reduction. Halo rows and the two-element global norm buffer are passed directly to CUDA-aware MPI.

The local reduction follows the same 256-thread, four-blocks-per-SM strategy as the Numba version. A custom kernel produces compact FP64 partial arrays, and `cp.sum` completes the local reductions before a CUDA-aware `MPI.Allreduce` combines the two values globally.

```text
CUDA-aware MPI halo exchange
             |
             v
CuPy RawKernel Jacobi update
             |
             v
RawKernel partial reduction
             |
             v
CuPy reduction of partial arrays
             |
             v
CUDA-aware MPI_Allreduce on device buffer
```

The current CuPy stream is synchronized before MPI reads or writes device buffers. The RawKernels require contiguous C-order FP64 arrays and use fixed-width NumPy scalar arguments matching their CUDA C++ signatures.

### `laplace_cupy_nccl.py`

This version uses CuPy RawKernels with NCCL point-to-point halo exchange and NCCL all-reduce. MPI remains responsible for launch, rank discovery, maximum-rank timing aggregation, and parallel HDF5 coordination.

Two non-blocking streams and two CUDA events expose communication and computation overlap:

```text
comm_stream:     wait(u_ready) -> NCCL halos -> record(halos_ready)

compute_stream:  deep-interior Jacobi
                           |
                           v
                 wait(halos_ready) -> interface Jacobi
```

`u_ready_event` prevents the next halo exchange from reading a field before the previous iteration has finished producing it. `halos_ready_event` delays only the interface-row kernels, leaving the deep-interior update eligible to overlap with NCCL traffic.

The convergence path performs the local RawKernel and CuPy reductions on `compute_stream`, followed by an NCCL all-reduce over two device-resident FP64 values. The implementation requires a CuPy NCCL backend exposing the version-specific `send_recv()` convenience method. It checks this capability at startup and aborts with a diagnostic if the method is unavailable.

### Distributed execution

Run one MPI rank per GPU. For reproducible backend comparisons, use the same fixed iteration count, grid, physical dimensions, and boundary values:

```bash
mpirun -np 4 python laplace_numba_mpi.py \
  --nx 4096 --ny 4096 --fixed-iters 1000 --check-every 100 \
  --output results/laplace_numba_mpi.h5

mpirun -np 4 python laplace_cupy_mpi.py \
  --nx 4096 --ny 4096 --fixed-iters 1000 --check-every 100 \
  --output results/laplace_cupy_mpi.h5

mpirun -np 4 python laplace_cupy_nccl.py \
  --nx 4096 --ny 4096 --fixed-iters 1000 --check-every 100 \
  --output results/laplace_cupy_nccl.h5
```

On Slurm, use the site-supported MPI launcher and GPU-binding policy. A representative launch is:

```bash
srun --ntasks=4 --gpus-per-task=1 --gpu-bind=closest \
  python laplace_cupy_nccl.py \
  --nx 4096 --ny 4096 --fixed-iters 1000
```

For tolerance-based termination, omit `--fixed-iters` and use `--tol`, `--max-iters`, and `--check-every`.

### Parallel HDF5 output

Each distributed solver copies its owned physical slab to host memory and writes the corresponding global hyperslab using MPI-enabled h5py:

```python
with h5py.File(filename, "w", driver="mpio", comm=mpi_comm) as f:
    dset = f.create_dataset("u", shape=(ny, nx), dtype=local_host.dtype)
    dset[ystart:yend, :] = local_host
```

The output records geometry, boundary values, dtype, rank count, decomposition, backend, communication method, iteration count, and final relative update norm as dataset attributes. The programs reject a serial h5py build by checking `h5py.get_config().mpi`.

### `compare_h5.py`

The comparison utility validates that both files contain a two-dimensional `u` dataset with matching shape and dtype and that both fields contain only finite values. It reports full-field and interior error metrics:

- maximum absolute difference;
- RMS absolute difference;
- relative L2 difference;
- interior maximum absolute difference;
- interior relative L2 difference.

The final decision uses combined absolute and relative tolerances:

```python
close = np.allclose(candidate, reference, rtol=args.rtol, atol=args.atol)
```

Compare outputs produced with identical fixed iteration counts:

```bash
python compare_h5.py \
  results/laplace_cupy_mpi.h5 \
  results/laplace_numba_mpi.h5 \
  --rtol 1.0e-11 --atol 1.0e-12

python compare_h5.py \
  results/laplace_cupy_mpi.h5 \
  results/laplace_cupy_nccl.h5 \
  --rtol 1.0e-11 --atol 1.0e-12
```

Parallel reductions do not reproduce an identical summation order across Numba, CuPy, MPI, and NCCL. Small floating-point differences are therefore expected. Select tolerances according to the grid, iteration count, and comparison purpose.

### Profiling the distributed solvers

For the CUDA-aware MPI implementations, inspect:

- stream synchronization before MPI accesses device buffers;
- the two nearest-neighbor halo exchanges per iteration;
- Jacobi launch placement after halo completion;
- partial-reduction kernels and compact second-stage reductions;
- host versus device placement of the global two-value reduction;
- convergence-check synchronization at `--check-every` intervals;
- imbalance from remainder rows or boundary-only ranks.

For the NCCL implementation, inspect:

- separate non-blocking compute and communication streams;
- the dependency from `u_ready_event` to NCCL boundary-row reads;
- overlap between NCCL halo exchange and deep-interior Jacobi work;
- the `halos_ready_event` wait before interface updates;
- NCCL all-reduce after local norm reduction;
- reduced overlap when local slabs contain no deep-interior rows.

### Distributed correctness and portability notes

- CUDA-aware MPI support must be present in the MPI build and enabled in the active transport configuration.
- Synchronizing before an MPI call provides ordering but does not prove that the MPI implementation avoids host staging.
- Every rank must enter matching point-to-point and collective operations.
- NCCL point-to-point calls must be matched consistently across neighboring ranks.
- Fixed-iteration mode is preferable for performance and HDF5 comparisons because every backend executes the same number of updates.
- The CUDA-event and wall-clock intervals include halo communication and periodic convergence checks, so the reported GCell-updates/s is an effective distributed solver metric.
- Parallel HDF5 requires an h5py build linked against parallel HDF5 and the same MPI family used to launch the application.

## Requirements

The distributed examples additionally require:

- mpi4py and an MPI implementation with CUDA-aware communication support;
- NCCL and a compatible `cupyx.distributed` NCCL backend for `laplace_cupy_nccl.py`;
- parallel HDF5 and an MPI-enabled h5py build for distributed output;
- one usable GPU per MPI rank and a correct local-rank-to-device mapping.


Common requirements:

- Python 3
- NumPy
- HDF5 support through `h5py` when `--output` is used

GPU variants additionally require:

- a supported NVIDIA GPU;
- a compatible NVIDIA driver and CUDA software stack;
- CuPy for `laplace_cupy.py` and `laplace_cupy_raw.py`;
- Numba CUDA for `laplace_numba.py`.

CuPy is optional for the computational path in `laplace_numba.py`, but it is used for NVTX annotations when available.

Verify the environment with commands such as:

```bash
nvidia-smi
python -c "import numpy as np; print(np.__version__)"
python -c "import h5py; print(h5py.__version__)"
python -c "import cupy as cp; print(cp.__version__); cp.show_config()"
python -c "from numba import cuda; print(cuda.is_available())"
```

On an HPC system, first load the required CUDA and Python modules and activate the intended virtual environment.

## Common command-line options

All four programs accept the following options:

```text
--nx NX
--ny NY
--lx LX
--ly LY
--max-iters MAX_ITERS
--tol TOL
--check-every CHECK_EVERY
--bc-left VALUE
--bc-right VALUE
--bc-bottom VALUE
--bc-top VALUE
--fixed-iters N
--output FILE
```

The GPU implementations also accept:

```text
--device DEVICE_ID
```

Default settings:

```text
nx             = 1024
ny             = 1024
lx             = 1.0
ly             = 1.0
max-iters      = 20000
tol            = 1.0e-8
check-every    = 100
bc-left        = 0.0
bc-right       = 1.0
bc-bottom      = 0.0
bc-top         = 0.0
device         = 0
fixed-iters    = disabled
output         = disabled
```

Display program-specific help with:

```bash
python laplace_numpy.py --help
python laplace_cupy.py --help
python laplace_numba.py --help
python laplace_cupy_raw.py --help
```

## Quick start

Run the NumPy reference:

```bash
python laplace_numpy.py \
    --nx 1024 \
    --ny 1024 \
    --tol 1.0e-8 \
    --max-iters 20000 \
    --check-every 100
```

Run the vectorized CuPy solver:

```bash
python laplace_cupy.py \
    --nx 1024 \
    --ny 1024 \
    --tol 1.0e-8 \
    --max-iters 20000 \
    --check-every 100 \
    --device 0
```

Run the Numba CUDA solver:

```bash
python laplace_numba.py \
    --nx 1024 \
    --ny 1024 \
    --tol 1.0e-8 \
    --max-iters 20000 \
    --check-every 100 \
    --device 0
```

Run the CuPy RawKernel solver:

```bash
python laplace_cupy_raw.py \
    --nx 1024 \
    --ny 1024 \
    --tol 1.0e-8 \
    --max-iters 20000 \
    --check-every 100 \
    --device 0
```

## Execution modes

### Tolerance mode

Tolerance mode is the default. The solver stops when the monitored relative update norm is smaller than `--tol`, or when `--max-iters` is reached.

Example:

```bash
python laplace_cupy_raw.py \
    --nx 2048 \
    --ny 1536 \
    --lx 2.0 \
    --ly 1.0 \
    --tol 1.0e-9 \
    --max-iters 50000 \
    --check-every 100 \
    --device 0
```

If the requested tolerance is not reached, the program prints a warning after completing `--max-iters` iterations.

### Fixed-iteration mode

Use `--fixed-iters` to execute exactly the requested number of Jacobi iterations:

```bash
python laplace_numba.py \
    --nx 4096 \
    --ny 4096 \
    --fixed-iters 5000 \
    --check-every 100 \
    --device 0
```

In this mode:

- tolerance-based termination is disabled;
- `--max-iters` does not determine the loop length;
- the norm is still evaluated according to `--check-every` and on the final iteration;
- the final reported norm is diagnostic only.

Fixed-iteration mode is recommended for performance comparisons and backend-to-backend validation because every implementation executes the same number of stencil updates.

## Reproducible cross-backend validation

Use identical geometry, physical dimensions, boundaries, iteration count, and norm-check frequency for every backend.

A compact validation workflow is:

```bash
mkdir -p results

python laplace_numpy.py \
    --nx 257 --ny 193 \
    --lx 2.0 --ly 1.0 \
    --fixed-iters 1000 \
    --check-every 100 \
    --output results/laplace_numpy.h5

python laplace_cupy.py \
    --nx 257 --ny 193 \
    --lx 2.0 --ly 1.0 \
    --fixed-iters 1000 \
    --check-every 100 \
    --device 0 \
    --output results/laplace_cupy.h5

python laplace_numba.py \
    --nx 257 --ny 193 \
    --lx 2.0 --ly 1.0 \
    --fixed-iters 1000 \
    --check-every 100 \
    --device 0 \
    --output results/laplace_numba.h5

python laplace_cupy_raw.py \
    --nx 257 --ny 193 \
    --lx 2.0 --ly 1.0 \
    --fixed-iters 1000 \
    --check-every 100 \
    --device 0 \
    --output results/laplace_cupy_raw.h5
```

The non-multiple dimensions `257 x 193` exercise the GPU bounds checks.

Compare the stored fields with Python:

```python
import h5py
import numpy as np

files = {
    "numpy": "results/laplace_numpy.h5",
    "cupy": "results/laplace_cupy.h5",
    "numba": "results/laplace_numba.h5",
    "cupy_raw": "results/laplace_cupy_raw.h5",
}

solutions = {}
for name, filename in files.items():
    with h5py.File(filename, "r") as h5:
        solutions[name] = h5["u"][...]

reference = solutions["numpy"]
for name, solution in solutions.items():
    diff = solution - reference
    print(
        name,
        "allclose=", np.allclose(solution, reference, rtol=1e-11, atol=1e-12),
        "max_abs=", np.max(np.abs(diff)),
        "relative_l2=", np.linalg.norm(diff) / max(np.linalg.norm(reference), 1e-300),
    )
```

The example tolerances are suitable starting points for these FP64 implementations, not universal mathematical guarantees.

## Boundary-condition example

Run with non-default boundary values:

```bash
python laplace_cupy_raw.py \
    --nx 1024 \
    --ny 768 \
    --lx 2.0 \
    --ly 1.0 \
    --bc-left 0.0 \
    --bc-right 1.0 \
    --bc-bottom -0.5 \
    --bc-top 0.5 \
    --fixed-iters 5000 \
    --device 0 \
    --output results/custom_bc.h5
```

The corners take the bottom and top values because those boundaries are assigned last.

## Warm-up and reset

Every implementation performs a warm-up before the measured run.

The warm-up exercises the Jacobi update and convergence calculation so that one-time setup is not attributed to the main solver interval. For GPU variants this includes operations such as runtime initialization, kernel compilation, initial launches, and reduction setup.

After warm-up, both solution fields are reset to the same zero-interior initial condition with the requested Dirichlet boundaries. The timed run therefore starts from an equivalent state in all four programs.

Warm-up time and reset time are not included in the reported solver interval.

## Timing methodology

### NumPy

The NumPy solver reports wall-clock time measured with `perf_counter()` around the complete monitored solver loop.

### GPU implementations

The three GPU implementations report two intervals:

- **CUDA-event time** measured by events recorded around the solver loop;
- **wall time** measured with `perf_counter()` around the same region.

The solver interval includes:

- all Jacobi updates;
- update-norm calculations performed at the selected check points;
- device-to-host scalar synchronization needed for Python convergence decisions;
- ping-pong swaps and loop control insofar as they affect the measured wall interval.

The interval excludes:

- object construction;
- persistent field allocation;
- initial boundary setup;
- warm-up;
- post-warm-up reset;
- optional HDF5 output.

The CUDA-event and wall times answer different questions. Event time primarily reflects work and synchronization on the CUDA stream. Wall time captures the host-observed duration of the monitored solver, including Python control overhead and scalar synchronization costs.

## Throughput metric

Each Jacobi iteration updates:

```text
(nx - 2) * (ny - 2)
```

physical interior cells. Effective throughput is reported as:

```text
GCell-updates/s =
    ((nx - 2) * (ny - 2) * niters) / elapsed_seconds / 1.0e9
```

This is labelled an **effective monitored solver throughput** because the elapsed interval includes periodic convergence checks, not only the stencil-update kernel.

The metric does not count the separate memory operations or floating-point operations performed by norm evaluation. Consequently, runs with different `--check-every` values are not directly comparable as pure Jacobi-kernel measurements.

For the GPU variants, throughput is reported using both CUDA-event and wall-clock elapsed time.

## Effect of `--check-every`

Norm evaluation is relatively expensive because it requires reductions and a host-visible scalar. Therefore:

```text
--check-every 1
```

provides fine stopping resolution but adds a convergence check and synchronization every iteration.

A larger value such as:

```text
--check-every 100
```

reduces monitoring overhead but allows tolerance mode to run up to 99 extra iterations after the first unchecked point at which the tolerance might have been satisfied.

For fair performance comparisons, all backends must use the same `--check-every` value.

For isolated stencil-kernel performance studies, use a fixed iteration count, a large check interval, and a profiler. Remember that the implementations always check at iteration 1 and at the final iteration.

## HDF5 output

Pass `--output` to write the final global solution:

```bash
python laplace_numpy.py \
    --nx 512 \
    --ny 384 \
    --fixed-iters 2000 \
    --output results/laplace_numpy.h5
```

The output directory is created automatically when needed.

Each file contains the dataset:

```text
/u
```

with shape:

```text
(ny, nx)
```

The dataset attributes record metadata including:

- `nx`, `ny`;
- `lx`, `ly`;
- `dx`, `dy`;
- all four boundary values;
- solver name;
- backend name;
- completed iteration count;
- final relative update norm;
- dtype;
- decomposition description;
- convergence-metric description;
- reduction method where applicable.

Inspect a file with:

```bash
h5ls -r results/laplace_numba.h5
h5dump -A results/laplace_numba.h5
```

Or use Python:

```python
import h5py

with h5py.File("results/laplace_numba.h5", "r") as h5:
    u = h5["u"]
    print("shape:", u.shape)
    print("dtype:", u.dtype)
    for key, value in u.attrs.items():
        print(f"{key}: {value}")
```

The GPU implementations explicitly copy the final field to host memory before serial HDF5 output.

## Memory usage

Each persistent solution field requires:

```text
ny * nx * 8 bytes
```

because all implementations use FP64. The two ping-pong fields therefore require approximately:

```text
2 * ny * nx * 8 bytes
```

The NumPy and vectorized CuPy variants can require additional temporary arrays created by stencil and reduction expressions.

The Numba CUDA and CuPy RawKernel variants allocate compact partial-reduction arrays. Their explicit partial workspace is approximately:

```text
2 * number_of_reduction_blocks * 8 bytes
```

This does not include runtime, compiler, allocator, context, or library overhead.

For a `4096 x 4096` grid, one FP64 field is approximately 128 MiB and the two persistent fields are approximately 256 MiB.

## NVTX profiling

The GPU programs emit NVTX ranges when CuPy NVTX support is available. In the Numba program, NVTX becomes a no-op if CuPy is unavailable.

The ranges identify phases such as:

- boundary initialization;
- warm-up;
- Jacobi updates;
- convergence-norm evaluation;
- the monitored solver loop;
- HDF5 output.

Profile a CuPy RawKernel run with Nsight Systems:

```bash
nsys profile \
    --trace=cuda,nvtx,osrt \
    --stats=true \
    --output=laplace_cupy_raw \
    python laplace_cupy_raw.py \
        --nx 4096 \
        --ny 4096 \
        --fixed-iters 1000 \
        --check-every 100 \
        --device 0
```

Profile the Numba kernels with Nsight Compute:

```bash
ncu \
    --set full \
    --target-processes all \
    --kernel-name-base demangled \
    --output=laplace_numba \
    python laplace_numba.py \
        --nx 4096 \
        --ny 4096 \
        --fixed-iters 100 \
        --check-every 100 \
        --device 0
```

Profiler metric collection can replay kernels and substantially alter runtime. Keep profiling runs separate from timing runs used for final performance comparisons.

## Running under Slurm

The following generic script runs all four implementations with a fixed iteration count. Adapt account, partition, GPU request, modules, and environment paths to the target system.

```bash
#!/bin/bash
#SBATCH --job-name=laplace_compare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=laplace_%j.out
#SBATCH --error=laplace_%j.err

set -euo pipefail

module purge
module load cuda
source /path/to/venv/bin/activate

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

mkdir -p results
nvidia-smi

COMMON=(
    --nx 2048
    --ny 1536
    --lx 2.0
    --ly 1.0
    --fixed-iters 5000
    --check-every 100
)

srun python laplace_numpy.py \
    "${COMMON[@]}" \
    --output results/numpy_${SLURM_JOB_ID}.h5

srun python laplace_cupy.py \
    "${COMMON[@]}" \
    --device 0 \
    --output results/cupy_${SLURM_JOB_ID}.h5

srun python laplace_numba.py \
    "${COMMON[@]}" \
    --device 0 \
    --output results/numba_${SLURM_JOB_ID}.h5

srun python laplace_cupy_raw.py \
    "${COMMON[@]}" \
    --device 0 \
    --output results/cupy_raw_${SLURM_JOB_ID}.h5
```

Some clusters expose only one device inside the job allocation and remap it to local device ID `0`. Follow the local scheduler and CUDA visibility policy.

For clean GPU timing, it is often preferable to run each GPU backend in a separate job or separate exclusive `srun` step.

## Suggested test matrix

### Minimal functional test

```bash
python laplace_numpy.py --nx 3 --ny 3 --fixed-iters 2 --check-every 1
```

### Non-multiple rectangular grid

```bash
python laplace_cupy_raw.py \
    --nx 259 \
    --ny 197 \
    --fixed-iters 100 \
    --check-every 10 \
    --device 0
```

### Anisotropic spacing

```bash
python laplace_numba.py \
    --nx 1025 \
    --ny 513 \
    --lx 4.0 \
    --ly 1.0 \
    --fixed-iters 1000 \
    --check-every 100 \
    --device 0
```

### Distinct boundary values

```bash
python laplace_cupy.py \
    --nx 512 \
    --ny 384 \
    --bc-left 1.0 \
    --bc-right 2.0 \
    --bc-bottom 3.0 \
    --bc-top 4.0 \
    --fixed-iters 500 \
    --check-every 50 \
    --device 0
```

### Tolerance-mode test

```bash
python laplace_numpy.py \
    --nx 256 \
    --ny 256 \
    --tol 1.0e-8 \
    --max-iters 50000 \
    --check-every 100
```

## Fair-comparison guidance

For meaningful backend comparisons:

1. Use exactly the same `nx`, `ny`, `lx`, `ly`, and boundary values.
2. Use fixed-iteration mode when comparing fields or throughput.
3. Use the same `--check-every` value because norm checks are inside the timed solver interval.
4. Run GPU programs on the same device under equivalent power, clock, and allocation conditions.
5. Keep warm-up enabled so first-use overhead is excluded consistently.
6. Compare wall time with wall time and CUDA-event time with CUDA-event time.
7. Do not interpret GCell-updates/s as isolated stencil throughput when convergence checks are included.
8. Use HDF5 output and compare the complete arrays, not only the final scalar norm.
9. Include non-square and non-block-multiple cases in correctness testing.
10. Record software versions, GPU model, grid dimensions, execution mode, and monitoring frequency.

## Reproducibility checklist

Record the following with benchmark results:

- GPU and CPU models;
- NVIDIA driver version;
- CUDA runtime and toolkit versions;
- Python version;
- NumPy, CuPy, Numba, Numba CUDA, and h5py versions;
- grid dimensions and physical domain size;
- boundary values and corner convention;
- fixed-iteration or tolerance mode;
- tolerance, maximum iterations, and check frequency;
- completed iterations and final update norm;
- CUDA-event and wall times;
- GPU block and reduction configuration;
- Slurm resource allocation and affinity settings;
- profiler use, if any.

Useful commands include:

```bash
nvidia-smi
python --version
python -c "import numpy; print('NumPy', numpy.__version__)"
python -c "import h5py; print('h5py', h5py.__version__)"
python -c "import cupy; print('CuPy', cupy.__version__); cupy.show_config()"
python -c "import numba; print('Numba', numba.__version__)"
```

## Expected performance characteristics

The NumPy implementation is the simplest correctness reference but operates on host memory and may allocate expression temporaries.

The vectorized CuPy implementation is concise and productive, but stencil expressions and reductions can require several GPU kernels and memory passes.

The Numba and CuPy RawKernel variants fuse the Jacobi update into one custom kernel and use compact partial-reduction workspaces. They are intended to reduce temporary allocation and unnecessary global-memory traffic relative to the vectorized expression form.

The Jacobi stencil is generally memory-bandwidth sensitive. Performance depends on grid size, memory-access efficiency, occupancy, reduction frequency, scalar synchronization, GPU architecture, and runtime/compiler behavior.

No implementation uses multi-GPU decomposition, MPI halo exchange, temporal blocking, asynchronous overlap, mixed precision, red-black ordering, multigrid acceleration, or CUDA Graph replay.

## Known limitations

- All implementations use FP64 only.
- The codes solve on a single global array and do not perform MPI domain decomposition.
- The convergence indicator is an iterate-update norm, not a discrete residual.
- The update norm is evaluated only at selected checkpoints.
- Tolerance-mode iteration counts can depend on `--check-every`.
- GPU norm checks copy a scalar to the host and synchronize execution.
- The reported throughput includes convergence-monitoring overhead.
- Vectorized NumPy and CuPy expressions may create temporary arrays.
- Custom CUDA kernels assume contiguous row-major storage.
- The Numba and RawKernel launch constants are source-level settings.
- HDF5 output is serial and requires a device-to-host copy for GPU results.
- The solver uses unaccelerated Jacobi iteration, which can converge slowly on fine grids.

## Troubleshooting

### CUDA is unavailable

Check the job allocation, driver visibility, loaded modules, and active Python environment:

```bash
nvidia-smi
echo "$CUDA_VISIBLE_DEVICES"
python -c "from numba import cuda; print(cuda.is_available())"
python -c "import cupy as cp; print(cp.cuda.runtime.getDeviceCount())"
```

### Device ID is out of range

Under schedulers, visible GPUs can be remapped. If only one GPU is visible, the correct local ID is usually `0` even when the physical GPU has another system-wide index.

### HDF5 import fails

Install or activate an environment containing `h5py`. HDF5 is imported only when output is requested.

### Runs stop later than expected

Tolerance is checked only at iteration 1, every `--check-every` iterations, and the final iteration. Reduce `--check-every` for finer stopping resolution.

### Backends produce different results

Confirm that all runs use the same:

- grid and physical dimensions;
- boundary values;
- corner assignment convention;
- fixed iteration count;
- initial condition;
- FP64 dtype.

Then compare the complete HDF5 fields and inspect maximum absolute and relative L2 differences.

### GPU wall time exceeds CUDA-event time

This is expected when Python overhead, scalar transfers, synchronization, and host control contribute appreciably to the monitored solver. Use both measurements and state clearly which one is being compared.
