# Introductory GPU Programming Examples with NumPy, CuPy, and Numba CUDA

This directory contains a progressive set of introductory examples for moving Python numerical workloads from CPU execution with NumPy to GPU execution with CuPy and Numba CUDA.
The examples cover array operations, host-device transfers, CUDA execution grids, grid-stride loops, reductions, kernel fusion, custom kernels, timing, correctness checks, NVTX annotations, and profiling with NVIDIA Nsight Systems.

The supplied Slurm script runs every example on one GPU of the **Leonardo Booster** partition and writes a separate Nsight Systems report for each program.

## Directory layout

```text
.
|-- intro_cupy_builtin.py
|-- intro_cupy_datatransfer.py
|-- intro_cupy_elementwise.py
|-- intro_cupy_fused.py
|-- intro_cupy_nonfused.py
|-- intro_cupy_rawkernel.py
|-- intro_cupy_simple.py
|-- intro_numba_2d.py
|-- intro_numba_gridstride.py
|-- intro_numba_reduction.py
|-- intro_numba_simple.py
|-- intro_numpy_kerneldesign.py
|-- intro_numpy_simple.py
|-- logs/
|   |-- run_intro_<jobid>.err
|   `-- run_intro_<jobid>.out
|-- profiles_<jobid>/
|   `-- nsys_*.job<jobid>.nsys-rep
`-- submit.sh
```

The `logs/` and `profiles_<jobid>/` directories are generated when the Slurm job is submitted. 
They are runtime artifacts rather than source files.

## Learning path

The examples are intended to be read approximately in the following order:

1.  `intro_numpy_simple.py`
2.  `intro_cupy_simple.py`
3.  `intro_cupy_datatransfer.py`
4.  `intro_cupy_builtin.py`
5.  `intro_cupy_nonfused.py`
6.  `intro_cupy_fused.py`
7.  `intro_cupy_elementwise.py`
8.  `intro_numpy_kerneldesign.py`
9.  `intro_cupy_rawkernel.py`
10. `intro_numba_simple.py`
11. `intro_numba_2d.py`
12. `intro_numba_gridstride.py`
13. `intro_numba_reduction.py`

This sequence starts with NumPy and CuPy array syntax, introduces explicit CUDA kernels with Numba, and then moves to kernel design, fusion and custom CuPy kernels.

## Example overview

### NumPy examples

#### `intro_numpy_simple.py`

Runs a simple vector expression on the CPU:

```python
h_y = np.sin(h_x * np.pi / 2.0)**2
```

It introduces the host-array naming convention and verifies the final value with `numpy.testing.assert_allclose`.

#### `intro_numpy_kerneldesign.py`

Expresses one Jacobi stencil update as an explicit point operation over a flattened two-dimensional array. 
The nested Python loops are intentionally simple and mirror the indexing logic later used by the CUDA `RawKernel` example.

The example highlights:

- conversion between two-dimensional coordinates and a one-dimensional offset;
- physical-interior and boundary handling;
- gathering left, right, down, and up neighbors;
- separation of input and output arrays.

This program is primarily a kernel-design reference. The Python loops are not intended to be performant.

### Basic CuPy examples

#### `intro_cupy_simple.py`

Ports the first NumPy expression to the GPU by replacing NumPy with CuPy. 
Arrays named with the `d_` prefix reside in device memory.

This example demonstrates the high degree of API compatibility between NumPy and CuPy for common array operations.

#### `intro_cupy_datatransfer.py`

Demonstrates explicit data movement between host and device memory:

- `cp.asarray(h_x)` copies a NumPy array to the GPU;
- `d_array.get()` copies a CuPy array to the host;
- `cp.asnumpy(d_array)` also returns a NumPy host array.

The program checks the trigonometric identity

```text
sin(x)^2 + cos(x)^2 = 1
```

on the host after transferring both GPU results back to CPU memory.

#### `intro_cupy_builtin.py`

Shows several built-in CuPy operations:

- element-wise arithmetic and transcendental functions;
- reductions;
- sorting;
- reshaping;
- a Frobenius norm through `cp.linalg.norm`;
- matrix multiplication through the `@` operator.

The example also checks both numerical results and output shape.

### CuPy expression fusion and custom kernels

#### `intro_cupy_nonfused.py`

Implements a two-dimensional Jacobi update using a regular Python function composed of CuPy array expressions. 
Because the expression is not fused, CuPy may launch multiple element-wise kernels and create intermediate results.

The cached execution is timed with CUDA events and marked with an NVTX range named `Non-fused Jacobi expression`.

The array size is deliberately large, so this example requires substantial GPU memory.

#### `intro_cupy_fused.py`

Uses `@cp.fuse` to fuse the element-wise Jacobi expression into a single generated kernel:

```python
@cp.fuse(kernel_name="fused_jacobi_expression")
def fused_jacobi_expression(left, right, down, up):
    return 0.25 * (left + right + down + up)
```

The first invocation performs JIT compilation. 
Compilation and initialization are excluded from the timed region through an explicit warm-up. 
The result is compared against an independent CuPy array-expression reference.

#### `intro_cupy_elementwise.py`

Introduces CuPy's higher-level custom-kernel interfaces:

- `cp.ElementwiseKernel` for element-by-element array addition;
- `cp.ReductionKernel` for an FP64-accumulated Frobenius norm.

Both kernels are warmed up before profiling. 
Their cached executions are enclosed in a single NVTX range, and the results are validated against standard CuPy operations.

#### `intro_cupy_rawkernel.py`

Implements a two-dimensional Jacobi stencil with CUDA C++ source embedded in Python and compiled through `cp.RawKernel`.

The custom kernel demonstrates:

- a two-dimensional CUDA grid and thread block;
- a `32 x 8` thread-block layout;
- offsets that map threads directly onto physical-interior points;
- bounds checks for both physical boundaries and padded grid coordinates;
- flattened row-major indexing;
- `const` and `__restrict__` qualified input access.

The high-level CuPy expression and cached `RawKernel` execution are placed in separate NVTX ranges. 
Their complete output arrays, including zero-valued boundaries, are compared for correctness.

### Numba CUDA examples

#### `intro_numba_simple.py`

Introduces an explicit one-dimensional Numba CUDA kernel for vector addition. It covers:

- `@cuda.jit` kernel declaration;
- `cuda.grid(1)` global indexing;
- bounds checking;
- host-to-device transfers with `cuda.to_device`;
- device output allocation;
- launch configuration with 256 threads per block;
- device-to-host transfer and verification.

The grid is rounded up to cover all elements, so some threads may exit through the bounds check.

#### `intro_numba_2d.py`

Extends the indexing model to a rectangular two-dimensional array. 
The kernel obtains `(column, row)` from `cuda.grid(2)` and uses a `16 x 16` thread block.

The example also demonstrates good timing practice:

1. launch and synchronize once to trigger compilation and initialization;
2. record CUDA events around the cached kernel launch;
3. synchronize the stop event;
4. report elapsed GPU time in milliseconds.

#### `intro_numba_gridstride.py`

Uses a one-dimensional grid-stride loop:

```python
start = cuda.grid(1)
stride = cuda.gridsize(1)
for i in range(start, d_c.size, stride):
    d_c[i] = d_a[i] + d_b[i]
```

Only 32,768 CUDA threads are launched for ten million elements. Each thread processes multiple elements by advancing with the total grid size. 
This pattern decouples grid size from problem size and is useful for persistent or occupancy-oriented launch configurations.

#### `intro_numba_reduction.py`

Defines a sum reduction with `@cuda.reduce`, applies it to a device array, and compares the scalar result with an FP64 NumPy reference.

This example provides a concise introduction to Numba's built-in reduction facility without requiring a handwritten block-reduction kernel.

## Naming conventions

The source uses prefixes to make memory placement explicit:

- `h_`: host-resident NumPy data;
- `d_`: device-resident CuPy or Numba CUDA data.

Other recurring conventions include:

- explicit warm-up before measuring JIT-compiled kernels;
- CUDA events for device execution timing;
- synchronization only where required for timing, dependencies, or host access;
- correctness checks outside profiled regions where practical;
- NVTX ranges for identifying operations in profiling timelines.

## Requirements

The examples require:

- a CUDA-capable NVIDIA GPU;
- a compatible NVIDIA driver and CUDA runtime;
- Python 3;
- NumPy;
- CuPy built for the active CUDA version;
- Numba with CUDA support;
- NVIDIA Nsight Systems for generation and inspection of `.nsys-rep` reports.

The included `submit.sh` is configured for the Leonardo Booster environment and loads CUDA 12.2, Python 3.11.7, GCC 12.2, Open MPI, NCCL, GDRCopy, and cuDNN modules. 
It activates the virtual environment at:

```text
${BASE_DIR}/pygpu_cuda122
```

The script expects the `intro/` directory and `pygpu_cuda122/` virtual environment to have the same parent directory because it sets `BASE_DIR` by temporarily moving to `..`.

## Running individual examples

After activating a compatible Python environment on a GPU node, run an example with:

```bash
python intro_cupy_simple.py
python intro_numba_2d.py
python intro_cupy_rawkernel.py
```

Most examples print:

```text
Verification passed.
```

Some also report elapsed kernel time or an execution-completion message.

The GPU examples must be run on a compute node with an allocated GPU rather than directly on a login node.

## Running the complete suite on Leonardo

Submit all examples with:

```bash
sbatch submit.sh
```

The job requests:

- one node;
- one task;
- 32 CPU cores;
- one GPU;
- exclusive node access;
- the `boost_usr_prod` partition;
- the `boost_qos_dbg` quality of service;
- a 30-minute time limit;
- all available node memory through `--mem=0`.

After submission, monitor the job with:

```bash
squeue -j <jobid>
```

Inspect accounting information after completion with:

```bash
sacct -j <jobid> --format=JobID,JobName,State,Elapsed,ExitCode
```

## Output and profiling reports

Standard output and standard error are written to:

```text
logs/run_intro_<jobid>.out
logs/run_intro_<jobid>.err
```

A successful run creates one Nsight Systems report per example under:

```text
profiles_<jobid>/
```

For example:

```text
profiles_<jobid>/nsys_intro_cupy_rawkernel.job<jobid>.nsys-rep
profiles_<jobid>/nsys_intro_numba_2d.job<jobid>.nsys-rep
```

The profiling command uses:

```bash
nsys profile \
  --force-overwrite=true \
  --sample=none \
  --cpuctxsw=none \
  --cuda-memory-usage=true \
  --trace=cuda,nvtx,osrt
```

This configuration records CUDA activity, NVTX annotations, selected operating-system runtime activity, and CUDA memory usage. 
CPU instruction-pointer sampling and CPU context-switch tracing are disabled to reduce unrelated profiling overhead.

Open a report in the Nsight Systems graphical interface with:

```bash
nsys-ui profiles_<jobid>/<report>.nsys-rep
```

On a system without the GUI, obtain a command-line summary with:

```bash
nsys stats profiles_<jobid>/<report>.nsys-rep
```

## What to inspect in Nsight Systems

Useful observations include:

- **NumPy example:** CPU-only execution and the absence of application CUDA kernels.
- **First CuPy or Numba operation:** initialization, memory allocation, context creation, and JIT-compilation overhead.
- **Warm versus cached launches:** compilation is excluded from the explicitly timed or NVTX-marked cached regions.
- **Host-device transfers:** CUDA memory copies associated with `cp.asarray`, `.get()`, `cp.asnumpy`, `cuda.to_device`, and `.copy_to_host()`.
- **Non-fused versus fused expressions:** differences in kernel count, intermediate operations, and launch overhead.
- **RawKernel execution:** one custom stencil kernel compared with the high-level CuPy expression.
- **Synchronization:** gaps or host waits caused by explicit synchronization and host data access.

## Resource considerations

Several examples intentionally allocate large arrays or execute expensive kernels to make behavior visible in a profiler.

In particular, `intro_cupy_nonfused.py` and `intro_cupy_fused.py` use `10000 x 10000` FP64 arrays. 
Each full array occupies approximately 800 MB, excluding allocator bookkeeping and temporary arrays. 
The non-fused expression may require additional temporary storage. 
Run these cases on a GPU with sufficient free memory.

## Reproducibility and correctness

The examples favor clarity over full benchmark infrastructure. Keep the following points in mind:

- most random initializations are not seeded, so input values may differ between runs;
- correctness checks use tolerances appropriate to the selected data type and operation;
- CUDA work is asynchronous unless a synchronization, event wait, or host transfer establishes completion;
- JIT compilation and library initialization can dominate a first invocation;
- device-event timings measure GPU work, whereas end-to-end application time also includes Python overhead, allocation, compilation, and data movement;
- profiling adds overhead and should not be treated as zero-overhead performance measurement.

## Cleaning generated files

Remove the logs and reports for a particular job with:

```bash
rm -f logs/run_intro_<jobid>.out logs/run_intro_<jobid>.err
rm -rf profiles_<jobid>
```

To remove all generated reports and logs from this directory:

```bash
rm -f logs/run_intro_*.out logs/run_intro_*.err
rm -rf profiles_[0-9]*
```

Review wildcard expansions before removing files in a shared or important working directory.

## Notes on `submit.sh`

The submission script uses `set -euo pipefail`, so it stops at the first failing command. 
Consequently, a failed example prevents later examples from running and normally leaves the corresponding error in `logs/run_intro_<jobid>.err`.

The script also:

- selects the GCC 12.2 `libstdc++` and `libgcc` runtime libraries;
- discovers the active CUDA installation from `nvcc`;
- configures a user-built parallel HDF5 installation;
- enables the CUB backend through `CUPY_ACCELERATORS=cub`;
- stores compiled CuPy kernels in a persistent cache;
- configures Open MPI to use UCX with CUDA-aware transports.

Some loaded modules and environment variables are broader than these introductory single-GPU examples require. 
They keep the runtime consistent with the larger GPU and MPI software environment used on the system.

## Scope

These programs are educational building blocks. 
They demonstrate individual CUDA Python concepts in small, inspectable scripts and provide timelines suitable for guided profiling exercises. 
They are not intended to replace application-level benchmarking, systematic parameter sweeps, or production solver implementations.

