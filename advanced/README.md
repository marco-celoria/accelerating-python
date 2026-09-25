# Advanced GPU Programming with CuPy, Numba CUDA, and PyTorch

This directory contains advanced CUDA Python examples built with CuPy, Numba CUDA, and PyTorch. The programs cover CUDA streams, asynchronous transfers, pinned host memory, cross-library interoperability, PyTorch autograd integration, JIT-compiled PyTorch C++/CUDA extensions, cooperative kernels, CUDA Graph capture and replay, FP64 WMMA tensor-core operations, NVTX annotation, and profiling with NVIDIA Nsight Systems and Nsight Compute.

The examples are listed in the same order in which `submit.sh` executes them.

## Directory layout

```text
.
|-- advanced_cupy_asynch.py
|-- advanced_cupy_cg.py
|-- advanced_cupy_graph.py
|-- advanced_cupy_numba.py
|-- advanced_cupy_streams.py
|-- advanced_cupy_tensorcore.py
|-- advanced_cupy_torch.py
|-- advanced_numba_asynch.py
|-- advanced_numba_cg.py
|-- advanced_numba_cupy.py
|-- advanced_numba_streams.py
|-- advanced_torch_custom.py
|-- logs/
|   |-- run_advanced_<jobid>.err
|   `-- run_advanced_<jobid>.out
|-- profiles_<jobid>/
|   |-- ncu_advanced_cupy_tensorcore.job<jobid>.ncu-rep
|   |-- nsys_advanced_cupy_asynch.job<jobid>.nsys-rep
|   |-- nsys_advanced_cupy_cg.job<jobid>.nsys-rep
|   |-- nsys_advanced_cupy_graph.job<jobid>.nsys-rep
|   |-- nsys_advanced_cupy_numba.job<jobid>.nsys-rep
|   |-- nsys_advanced_cupy_streams.job<jobid>.nsys-rep
|   |-- nsys_advanced_cupy_tensorcore.job<jobid>.nsys-rep
|   |-- nsys_advanced_cupy_torch.job<jobid>.nsys-rep
|   |-- nsys_advanced_numba_asynch.job<jobid>.nsys-rep
|   |-- nsys_advanced_numba_cg.job<jobid>.nsys-rep
|   |-- nsys_advanced_numba_cupy.job<jobid>.nsys-rep
|   |-- nsys_advanced_numba_streams.job<jobid>.nsys-rep
|   `-- nsys_advanced_torch_custom.job<jobid>.nsys-rep
|-- README.md
`-- submit.sh
```

The `logs/` and `profiles_<jobid>/` directories are generated at runtime. The profile directory contains the Nsight Systems and Nsight Compute reports produced by the job.

## Execution and presentation order

The examples are run and documented in this order:

1. `advanced_cupy_streams.py`
2. `advanced_numba_streams.py`
3. `advanced_cupy_asynch.py`
4. `advanced_numba_asynch.py`
5. `advanced_cupy_numba.py`
6. `advanced_numba_cupy.py`
7. `advanced_cupy_torch.py`
8. `advanced_torch_custom.py`
9. `advanced_cupy_cg.py`
10. `advanced_numba_cg.py`
11. `advanced_cupy_graph.py`
12. `advanced_cupy_tensorcore.py`

The final tensor-core program is executed twice, first under Nsight Systems and then under Nsight Compute.

## Example overview

### 1. `advanced_cupy_streams.py`

Demonstrates CUDA stream scheduling and event dependencies with CuPy. Three deliberately compute-intensive `cp.ElementwiseKernel` objects repeatedly evaluate sine, exponential, and square-root recurrences so that their execution is clearly visible in an Nsight Systems timeline.

The program creates three custom blocking streams:

```python
stream1 = cp.cuda.Stream(non_blocking=False)
stream2 = cp.cuda.Stream(non_blocking=False)
stream3 = cp.cuda.Stream(non_blocking=False)
```

Because these are blocking streams, they retain legacy synchronization behavior with the CUDA NULL stream. The kernels are compiled and warmed up before the profiled execution.

The execution graph is:

```text
Operation 0: long_sqrt_kernel(x)     [NULL stream]

Operation 1: long_sin_kernel(x)      [stream 1, independent branch]

Operation 2: long_exp_kernel(x)      [stream 2]
                    |
                    | event: y2_ready
                    v
Operation 3: long_sqrt_kernel(y2)    [stream 3]
```

The `y2_ready` event is recorded after Operation 2. Stream 3 waits for that event before submitting the kernel that consumes `y2`. The program synchronizes stream 1 separately and then synchronizes stream 3, which also waits transitively for stream 2.

NVTX ranges identify each submission region in the profiler. The example is designed to expose stream ordering and dependencies rather than minimize total execution time.

### 2. `advanced_numba_streams.py`

Recreates the preceding stream topology with Numba CUDA kernels. It uses explicit one-dimensional kernels for the same sine, exponential, and square-root recurrences.

Unlike a CuPy `ElementwiseKernel`, a Numba CUDA kernel does not allocate and return its output. The example therefore preallocates four device arrays:

```python
d_y0 = cuda.device_array_like(d_x)
d_y1 = cuda.device_array_like(d_x)
d_y2 = cuda.device_array_like(d_x)
d_y3 = cuda.device_array_like(d_x)
```

The legacy default stream is selected explicitly:

```python
default_stream = cuda.legacy_default_stream()
```

Three ordinary streams are created with `cuda.stream()`. The stream is passed as the third item in each Numba launch configuration:

```python
long_sin_kernel[blocks_per_grid, threads_per_block, stream1](
    d_x, d_y1, iterations
)
```

The dependency graph matches the CuPy example:

```text
Operation 0: long_sqrt_kernel(d_x, d_y0)  [legacy default stream]

Operation 1: long_sin_kernel(d_x, d_y1)   [stream 1, independent branch]

Operation 2: long_exp_kernel(d_x, d_y2)   [stream 2]
                          |
                          | event: y2_ready
                          v
Operation 3: long_sqrt_kernel(d_y2, d_y3) [stream 3]
```

The event is recorded in stream 2 and made visible to stream 3 with:

```python
y2_ready.wait(stream3)
```

All kernels are first launched on a 1,024-element device view to trigger Numba JIT compilation before profiling. Device arrays, streams, events, and numerical kernels use Numba CUDA. CuPy is imported only for NVTX range annotations.

### 3. `advanced_cupy_asynch.py`

Demonstrates chunked asynchronous transfers and GPU computation with CuPy.

The problem contains one billion FP64 elements and is divided across four streams. Host input and output arrays are allocated in pinned memory with `cupyx.empty_pinned`. Pinned memory is required for genuinely asynchronous host-device transfers.

For each chunk, the program submits the following ordered operations to one non-blocking stream:

```text
Pinned host a and b
        |
        | asynchronous H2D copies
        v
Device a and b
        |
        | d_o = 0.5 * hypot(d_a, d_b)
        v
Device output
        |
        | asynchronous D2H copy
        v
Pinned host output slice
```

The four streams may overlap host-to-device transfers, kernel execution, and device-to-host transfers when the hardware and transfer directions permit it.

The call:

```python
d_o.get(out=h_o[start:stop], blocking=False)
```

enqueues the device-to-host transfer into the current stream without blocking the host. Explicit references to the device arrays are retained in `device_buffers` until all stream operations have completed, preventing premature release or reuse of the underlying allocations.

After every stream is synchronized, the result is compared with an independent NumPy reference.

### 4. `advanced_numba_asynch.py`

Implements the same four-stream, chunked asynchronous pipeline with Numba CUDA.

Pinned arrays are allocated with:

```python
h_a = cuda.pinned_array(n_elements, dtype=np.float64)
h_b = cuda.pinned_array(n_elements, dtype=np.float64)
h_o = cuda.pinned_array(n_elements, dtype=np.float64)
```

For each chunk, the program enqueues the following operations in the same stream:

1. asynchronous transfer of `h_a[start:stop]` to `d_a`;
2. asynchronous transfer of `h_b[start:stop]` to `d_b`;
3. allocation of the device output;
4. launch of `scaled_hypot_kernel`;
5. asynchronous copy of `d_o` into the corresponding pinned output slice.

Operations submitted to one stream execute in order, so no event is required inside a chunk. Separate streams allow independent chunks to make progress concurrently.

As in the CuPy version, device allocations are retained until all asynchronous work is complete. The final pinned output is checked against NumPy after synchronizing every stream.

### 5. `advanced_cupy_numba.py`

Demonstrates interoperability when CuPy owns the CUDA stream and Numba uses that same stream.

CuPy creates a non-blocking stream:

```python
cupy_stream = cp.cuda.Stream(non_blocking=True)
```

Numba wraps its native handle without taking ownership:

```python
numba_stream = cuda.external_stream(cupy_stream.ptr)
```

CuPy remains responsible for the stream lifetime. CuPy arrays are allocated and initialized on the shared stream, then passed directly to a Numba CUDA kernel. This works because both libraries expose compatible CUDA device-memory interfaces.

Dependent operations are submitted to the same stream:

```text
CuPy allocation and initialization
                |
                v
Numba vector-addition kernel
                |
                v
CuPy square root and reduction
```

No intermediate host copies are required. Stream ordering establishes the dependencies, and synchronization occurs only before the host consumes the scalar results.

### 6. `advanced_numba_cupy.py`

Demonstrates the reverse ownership model: Numba creates and owns the CUDA stream, while CuPy wraps it as an external stream.

```python
numba_stream = cuda.stream()
cupy_stream = cp.cuda.Stream.from_external(numba_stream)
```

The input is transferred to the GPU through Numba. CuPy then creates a zero-copy view of the same device allocation:

```python
d_x_cupy = cp.asarray(d_x_numba)
```

The CuPy operation modifies the Numba allocation in place. Two asynchronous device-to-host transfers are then queued in the same underlying CUDA stream, one through CuPy and one through Numba.

Because all operations share one stream, their order enforces the dependency chain. No manual event or intermediate synchronization is needed. The stream is synchronized only before the host compares both outputs with the NumPy reference.

### 7. `advanced_cupy_torch.py`

Integrates custom CuPy `RawKernel` operations with PyTorch autograd.

The example defines:

- a forward CUDA kernel computing `y = x^2 + 3x - 2`;
- a backward CUDA kernel computing `grad_x = grad_y * (2x + 3)`;
- a custom `torch.autograd.Function`;
- a small `torch.nn.Module` wrapper.

The forward and backward methods validate that tensors are CUDA-resident FP32 data and ensure contiguous storage. Empty tensors return immediately, while the number of elements is checked against `INT32_MAX` because the raw kernels receive `n` as a 32-bit integer.

PyTorch owns the active CUDA stream. CuPy wraps that stream so that custom kernels are correctly ordered with surrounding PyTorch work:

```python
torch_stream = torch.cuda.current_stream(x.device)
cupy_stream = cp.cuda.Stream.from_external(torch_stream)
```

DLPack creates zero-copy CuPy views of PyTorch tensors:

```python
cp.from_dlpack(x.detach())
cp.from_dlpack(y)
```

The custom implementation is validated against native PyTorch in both the forward pass and backward gradient calculation using `torch.testing.assert_close`.

### 8. `advanced_torch_custom.py`

Builds and executes a native PyTorch C++/CUDA extension with `torch.utils.cpp_extension.load_inline`, then integrates its forward and backward CUDA kernels with PyTorch autograd.

The extension implements the same elementwise function used by `advanced_cupy_torch.py`:

```text
Forward:
    y = x^2 + 3x - 2

Backward:
    grad_x = grad_y * (2x + 3)
```

The CUDA source contains two one-dimensional kernels:

- `forward_kernel`, which computes the forward transformation;
- `backward_kernel`, which applies the analytical derivative to the incoming gradient.

The extension is compiled and loaded at runtime with:

```python
custom_ext = load_inline(
    name="custom_torch_ext",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["custom_forward", "custom_backward"],
    verbose=False,
)
```

The C++ wrappers allocate outputs with `torch::empty_like`, return immediately for empty tensors, and launch 256 threads per block. Both kernels execute in PyTorch's current CUDA stream:

```cpp
cudaStream_t stream = at::cuda::getCurrentCUDAStream();
```

Using the current stream preserves ordering between the extension kernels and surrounding native PyTorch operations without an intermediate device synchronization.

The Python `TorchCustomFunction` validates that the forward input is a CUDA FP32 tensor, creates contiguous storage when needed, saves the contiguous input for the backward pass, and makes the incoming gradient contiguous before passing it to the extension:

```python
class TorchCustomFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        if not x.is_cuda or x.dtype != torch.float32:
            raise TypeError("Expected a CUDA float32 tensor")
        x = x.contiguous()
        ctx.save_for_backward(x)
        return custom_ext.custom_forward(x)

    @staticmethod
    def backward(ctx, grad_y):
        x, = ctx.saved_tensors
        grad_y = grad_y.contiguous()
        return custom_ext.custom_backward(x, grad_y)
```

A small `torch.nn.Module` wrapper exposes the operation through the standard module interface. The program evaluates 1,000 FP32 elements, computes a backward pass with an explicitly generated upstream gradient, and compares both the output and input gradient with a pure PyTorch reference using `torch.testing.assert_close`.

Successful execution prints:

```text
SUCCESS: Pure PyTorch inline CUDA forward and backward match reference
```

Before importing PyTorch, the source sets:

```python
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
```

This builds the extension for compute capability 8.0, which matches the NVIDIA A100 GPUs used on Leonardo Booster. The value must be changed or generalized when compiling for GPUs with a different compute capability.

The first execution may spend substantially more time compiling and linking the extension than running its small CUDA workload. Later executions can reuse PyTorch's extension cache when the source, extension name, software environment, compiler configuration, and target architecture remain compatible.

### 9. `advanced_cupy_cg.py`

Implements FP64 normalization in a single cooperative CuPy `RawKernel`:

```text
y[i] = x[i] / sum(x)
```

The CUDA C++ kernel performs five stages:

1. grid-stride accumulation with `double2` vector loads;
2. warp-level reduction using a 32-thread tiled partition;
3. block-level reduction through shared memory and one FP64 atomic addition per block;
4. a cooperative grid-wide barrier;
5. normalization using the completed global sum, again with `double2` accesses.

The kernel is compiled with cooperative-launch support:

```python
cooperative_normalize = cp.RawKernel(
    kernel_source,
    "cooperative_normalize",
    options=("--std=c++17",),
    enable_cooperative_groups=True,
)
```

Before launch, the program verifies device support for cooperative launches, the compiled kernel's thread-block limit, a positive occupancy result, C-contiguous inputs and outputs, and 16-byte alignment for `double2` accesses.

The number of blocks is derived from the occupancy API and the number of streaming multiprocessors. This is essential because every block in a cooperative launch must be concurrently resident for `grid.sync()` to be valid.

The global FP64 accumulator is zeroed before every launch. Results are checked against a CuPy reduction and vectorized normalization reference. The deliberately odd problem size exercises the scalar tail after `double2` vectorization.

### 10. `advanced_numba_cg.py`

Implements the same single-kernel FP64 normalization with Numba CUDA cooperative groups.

The kernel uses:

- `cuda.cg.this_grid()` for the grid-wide cooperative barrier;
- `cuda.shfl_down_sync` for warp reductions;
- static shared memory for warp partial sums;
- `cuda.atomic.add` for one contribution per block;
- grid-stride loops for accumulation and normalization.

Numba does not expose the same thread-block Cooperative Groups abstraction used by the CUDA C++ version, so `cuda.syncthreads()` supplies the required block-wide barrier.

The program first compiles a kernel specialization with a warm-up launch. It then queries the compiled overload for the reported maximum cooperative grid size.

A documented, stack-specific workaround caps the launch at five blocks per SM:

```python
VERIFIED_COOPERATIVE_BLOCKS_PER_SM = 5
```

This cap was verified for the software versions stated in the source because larger launches fail with `CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE` on that stack despite the higher value returned by the occupancy query. The selected launch size is checked against the reported maximum before execution.

CuPy provides the large device arrays, reference implementation, numerical assertions, and NVTX annotations. Numba CUDA provides the cooperative kernel. As in the CuPy implementation, the global accumulator must be initialized to zero before every launch.

### 11. `advanced_cupy_graph.py`

Demonstrates CUDA Graph stream capture and repeated replay with CuPy. The captured workload evaluates:

```text
tmp = sin(x)
out = cos(y)
out = tmp + out
```

A dedicated non-blocking stream is created for initialization, warm-up, capture, and replay:

```python
stream = cp.cuda.Stream(non_blocking=True)
```

Persistent input, output, and intermediate arrays are allocated before capture. The `out=` arguments ensure that the captured ufunc sequence writes into existing device buffers instead of creating result arrays:

```python
with stream:
    x = cp.ones((1000, 1000), dtype=cp.float32)
    y = cp.ones((1000, 1000), dtype=cp.float32)
    tmp = cp.empty_like(x)
    out = cp.empty_like(x)

    cp.sin(x, out=tmp)
    cp.cos(y, out=out)
    cp.add(tmp, out, out=out)
```

The warm-up executes the exact operations that will be captured. This moves lazy kernel compilation and first-use initialization outside the capture region. The stream is synchronized before capture begins.

The static operation sequence is then captured:

```python
with stream:
    stream.begin_capture()

    cp.sin(x, out=tmp)
    cp.cos(y, out=out)
    cp.add(tmp, out, out=out)

    graph = stream.end_capture()
```

The instantiated graph is replayed on the same stream:

```python
for _ in range(n_replays):
    graph.launch(stream=stream)

stream.synchronize()
```

Graph replay preserves the captured dependencies and reuses the same device addresses. The arrays must therefore remain alive while the graph can execute. Input values may be changed by writing into the existing `x` and `y` buffers, but rebinding either Python variable to a new array does not redirect the captured graph to a new allocation.

The primary benefit appears when the short sequence is replayed many times. Graph construction adds one-time overhead, while replay reduces repeated CPU-side submission overhead relative to launching the three kernels separately on every iteration.

Validation occurs only after replay and synchronization. The entire output is compared with a CuPy reference using combined relative and absolute tolerances:

```python
expected = cp.sin(x) + cp.cos(y)
cp.testing.assert_allclose(out, expected, rtol=1.0e-6, atol=1.0e-7)
```

Host transfers, scalar extraction, dynamic allocation, and other potentially capture-incompatible work are deliberately kept outside the capture region.

### 12. `advanced_cupy_tensorcore.py`

Demonstrates an FP64 WMMA matrix multiply-accumulate operation through a CuPy `RawKernel` and CUDA's `mma.h` API.

FP64 WMMA uses an `8 x 8 x 4` tile:

```text
A: 8 x 4
B: 4 x 8
C: 8 x 8
```

One warp of 32 threads loads the input fragments, initializes the accumulator fragment, performs `wmma::mma_sync`, and stores the FP64 result in row-major order.

The launch configuration is exactly one block containing one warp:

```python
fp64_wmma_kernel((1,), (32,), (d_a, d_b, d_c))
```

The result is compared with CuPy's high-level matrix multiplication. The submission script profiles this program twice:

- Nsight Systems records the CUDA timeline and launch activity;
- Nsight Compute collects the full kernel metric set.

This example requires a GPU architecture and CUDA toolchain that support FP64 WMMA operations.

## Requirements

The examples require:

- a CUDA-capable NVIDIA GPU;
- a compatible NVIDIA driver and CUDA runtime;
- Python 3;
- NumPy;
- CuPy built for the active CUDA version;
- Numba CUDA;
- PyTorch with CUDA support for `advanced_cupy_torch.py` and `advanced_torch_custom.py`;
- a compatible host C++ compiler, CUDA compiler, CUDA toolkit headers, and native build utilities for `advanced_torch_custom.py`;
- a writable PyTorch extension build/cache directory for the inline C++/CUDA extension;
- NVIDIA Nsight Systems;
- NVIDIA Nsight Compute.

The cooperative examples additionally require device support for cooperative kernel launches. The CUDA Graph example requires CUDA stream-capture support in the active CUDA runtime and CuPy build. The FP64 WMMA example requires hardware support for FP64 tensor-core operations.

`advanced_torch_custom.py` sets `TORCH_CUDA_ARCH_LIST=8.0` before importing PyTorch and invoking the extension build. The loaded host compiler, CUDA toolkit, PyTorch build, Python environment, and requested GPU architecture must be mutually compatible.

## Leonardo environment

The included `submit.sh` is configured for Leonardo Booster. It loads:

```text
gcc/12.2.0-binut2.41
cuda/12.2
python/3.11.7
openmpi/5.0.9--gcc--12.2.0-cuda-12.2
nccl/2.22.3-1--gcc--12.2.0-cuda-12.2-spack0.22
gdrcopy/2.4.1-cuda-12.2
cudnn/8.9.7.29-12--gcc--12.2.0-cuda-12.2
```

The script activates:

```text
${BASE_DIR}/pygpu_cuda122
```

It expects the `advanced/` directory and `pygpu_cuda122/` virtual environment to share the same parent directory, since `BASE_DIR` is obtained by temporarily moving to `..`.

The script also:

- selects the GCC 12.2 runtime libraries and preloads `libstdc++` and `libgcc_s`;
- discovers the active CUDA installation from `nvcc`;
- exports `CUDA_HOME`, `CUDA_PATH`, and `CUDACXX`;
- configures a user-built parallel HDF5 installation;
- enables the CUB CuPy accelerator;
- establishes a persistent CuPy kernel cache;
- configures CUDA-aware UCX transports for Open MPI.

The GCC and CUDA settings are also required by `advanced_torch_custom.py`, because `load_inline` invokes PyTorch's native C++/CUDA build pipeline at runtime. The source targets compute capability 8.0, matching the NVIDIA A100 GPUs used on Leonardo Booster.

The submission script does not explicitly set `TORCH_EXTENSIONS_DIR`. PyTorch therefore uses its default extension build/cache location. That location must be writable from the compute node. If a project-local persistent cache is preferred, the script can define one before running the examples:

```bash
export TORCH_EXTENSIONS_DIR="${BASE_DIR}/.torch_extensions"
mkdir -p "${TORCH_EXTENSIONS_DIR}"
```

A persistent cache can avoid rebuilding an unchanged extension in later jobs. Cached extensions may need to be removed after changes to the source, extension name, Python version, PyTorch version, CUDA toolkit, host compiler, compiler flags, or target GPU architecture.

Some loaded modules and environment variables are broader than these single-GPU examples require. They keep the runtime consistent with the larger CUDA, MPI, NCCL, and HDF5 environment used on Leonardo.

## Running individual examples

Run GPU examples only inside an allocation on a compute node. After loading the required modules and activating the Python environment, execute individual programs in the documented order:

```bash
python advanced_cupy_streams.py
python advanced_numba_streams.py
python advanced_cupy_asynch.py
python advanced_numba_asynch.py
python advanced_cupy_numba.py
python advanced_numba_cupy.py
python advanced_cupy_torch.py
python advanced_torch_custom.py
python advanced_cupy_cg.py
python advanced_numba_cg.py
python advanced_cupy_graph.py
python advanced_cupy_tensorcore.py
```

Programs that contain correctness checks print `Verification passed.`, `SUCCESS: Pure PyTorch inline CUDA forward and backward match reference`, or an equivalent success message after completing successfully.

The first invocation of `advanced_torch_custom.py` may compile the inline extension before running the CUDA workload. A later compatible invocation may reuse the cached extension.

## Running the complete suite

Submit the complete sequence with:

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
- all node memory through `--mem=0`.

Monitor the job with:

```bash
squeue -j <jobid>
```

Inspect the final job state with:

```bash
sacct -j <jobid> --format=JobID,JobName,State,Elapsed,ExitCode
```

The script uses `set -euo pipefail`. A failing example stops the suite immediately, so later programs and reports will not be produced after the first error.

The current `submit.sh` executes and profiles `advanced_torch_custom.py` immediately after `advanced_cupy_torch.py`, matching the order documented in this README.

## Profiling order and generated reports

Standard output and standard error are written to:

```text
logs/run_advanced_<jobid>.out
logs/run_advanced_<jobid>.err
```

The profiling reports are generated in this exact execution order:

```text
profiles_<jobid>/nsys_advanced_cupy_streams.job<jobid>.nsys-rep
profiles_<jobid>/nsys_advanced_numba_streams.job<jobid>.nsys-rep
profiles_<jobid>/nsys_advanced_cupy_asynch.job<jobid>.nsys-rep
profiles_<jobid>/nsys_advanced_numba_asynch.job<jobid>.nsys-rep
profiles_<jobid>/nsys_advanced_cupy_numba.job<jobid>.nsys-rep
profiles_<jobid>/nsys_advanced_numba_cupy.job<jobid>.nsys-rep
profiles_<jobid>/nsys_advanced_cupy_torch.job<jobid>.nsys-rep
profiles_<jobid>/nsys_advanced_torch_custom.job<jobid>.nsys-rep
profiles_<jobid>/nsys_advanced_cupy_cg.job<jobid>.nsys-rep
profiles_<jobid>/nsys_advanced_numba_cg.job<jobid>.nsys-rep
profiles_<jobid>/nsys_advanced_cupy_graph.job<jobid>.nsys-rep
profiles_<jobid>/nsys_advanced_cupy_tensorcore.job<jobid>.nsys-rep
profiles_<jobid>/ncu_advanced_cupy_tensorcore.job<jobid>.ncu-rep
```

The final extension shown for the Nsight Compute report is the standard report extension associated with an `ncu -o` output prefix.

### Nsight Systems configuration

Every example is profiled with:

```bash
nsys profile \
  --force-overwrite=true \
  --sample=none \
  --cpuctxsw=none \
  --cuda-memory-usage=true \
  --trace=cuda,nvtx,osrt
```

This captures CUDA activity, NVTX annotations, selected operating-system runtime activity, and CUDA memory usage. CPU sampling and CPU context-switch tracing are disabled to reduce unrelated profiler overhead.

Open a report graphically with:

```bash
nsys-ui profiles_<jobid>/<report>.nsys-rep
```

Generate a command-line summary with:

```bash
nsys stats profiles_<jobid>/<report>.nsys-rep
```

### Nsight Compute configuration

The tensor-core kernel is additionally profiled with:

```bash
ncu --set full \
  -o profiles_<jobid>/ncu_advanced_cupy_tensorcore.job<jobid> \
  python advanced_cupy_tensorcore.py
```

The `full` metric set provides detailed launch, scheduler, memory, instruction, occupancy, and compute-pipeline information. It introduces significant replay and collection overhead, which is acceptable for this small one-warp kernel but should be used selectively for larger applications.

Inspect the generated report with the Nsight Compute GUI or command-line interface:

```bash
ncu-ui profiles_<jobid>/ncu_advanced_cupy_tensorcore.job<jobid>.ncu-rep
```

```bash
ncu --import profiles_<jobid>/ncu_advanced_cupy_tensorcore.job<jobid>.ncu-rep \
  --page details
```

## What to inspect in the profilers

### Stream examples

For `advanced_cupy_streams.py` and `advanced_numba_streams.py`, inspect:

- warm-up and JIT-compilation activity before the annotated execution;
- the relationship between the legacy default stream and custom streams;
- the independent stream-1 branch;
- event recording in stream 2;
- the event wait before the stream-3 consumer;
- host-side submission ranges versus asynchronous device execution;
- final host waits caused by stream synchronization.

### Asynchronous transfer examples

For `advanced_cupy_asynch.py` and `advanced_numba_asynch.py`, inspect:

- pageable versus pinned memory behavior;
- H2D and D2H transfer activity;
- whether copies and kernels overlap across streams;
- whether one or two GPU copy engines are active in the timeline;
- chunk boundaries and operation ordering within each stream;
- serialization introduced by allocation, initialization, or host activity.

### CuPy and Numba interoperability

For `advanced_cupy_numba.py` and `advanced_numba_cupy.py`, inspect:

- that both libraries submit work to the same underlying CUDA stream;
- the absence of intermediate device-host-device copies;
- correct ordering across library boundaries;
- synchronization only when host-visible results are required;
- memory-ownership and stream-ownership differences between the two directions.

### PyTorch interoperability and custom extensions

For `advanced_cupy_torch.py`, inspect:

- PyTorch and CuPy work submitted to the same current CUDA stream;
- forward and backward CuPy raw-kernel launches;
- the absence of copies when DLPack views are created;
- ordering between custom CuPy kernels and native PyTorch operations;
- the autograd backward execution sequence.

For `advanced_torch_custom.py`, inspect:

- the native `forward_kernel` and `backward_kernel` CUDA launches;
- the 256-thread launch configuration and corresponding grid sizes;
- use of PyTorch's current CUDA stream by both extension kernels;
- ordering between the extension kernels and native PyTorch reference operations;
- output allocation performed through `torch::empty_like`;
- the transition from `TorchCustomFunction.forward` to `TorchCustomFunction.backward`;
- synchronization associated with final validation or other host-visible results;
- differences between a first run that builds the extension and a later run that reuses a compatible cache.

Compilation performed by `load_inline` is host-side build activity rather than CUDA kernel execution. Nsight Systems is principally useful for inspecting the CUDA work after the extension has been compiled and loaded. For a cleaner and more representative GPU timeline, build the extension once and reuse the cached build in the profiled execution.

### Cooperative kernels

For `advanced_cupy_cg.py` and `advanced_numba_cg.py`, inspect:

- cooperative-launch configuration;
- the number of blocks selected per SM;
- long-running grid-stride accumulation and normalization stages;
- synchronization effects around the grid-wide barrier;
- FP64 atomic activity;
- differences between the vectorized CuPy raw kernel and scalar Numba implementation;
- the separate CuPy reference computation used for validation.

### CUDA Graph capture and replay

For `advanced_cupy_graph.py`, inspect:

- initialization and warm-up activity before stream capture;
- the stream-capture and graph-instantiation API activity;
- the three dependent ufunc kernels represented by each graph launch;
- repeated graph launches with lower host-submission overhead than separate kernel submissions;
- the absence of output-array allocation inside the captured sequence;
- synchronization only after the replay loop;
- NVTX ranges that separate warm-up, capture, replay, and validation, if provided by the source.

When comparing ordinary launches with graph replay, distinguish graph-construction cost from steady-state replay cost. A single replay is not expected to amortize capture and instantiation overhead.

### FP64 WMMA kernel

For `advanced_cupy_tensorcore.py`, use Nsight Systems to inspect launch placement and duration. Use Nsight Compute to inspect:

- the one-block, one-warp launch configuration;
- achieved occupancy and why it is not a useful performance target for this minimal example;
- instruction mix and tensor-pipeline activity;
- global-memory loads and stores for the small matrices;
- warp execution and scheduler behavior;
- FP64 matrix-multiply-accumulate instructions generated from `wmma::mma_sync`.

## Memory and runtime considerations

Several examples deliberately use very large FP64 arrays.

### Asynchronous transfer examples

`advanced_cupy_asynch.py` and `advanced_numba_asynch.py` allocate three pinned host arrays containing one billion FP64 elements each. Each array occupies approximately 8 GB, for approximately 24 GB of pinned host memory before temporary NumPy reference results and allocator overhead.

The two input initializations and final NumPy reference can introduce substantial CPU time and additional host-memory pressure. Pinned memory is a limited operating-system resource, so these examples should be run only on a node where the requested allocation and memory-locking limits permit this footprint.

Each of the four chunks contains up to 250 million FP64 elements. The three device arrays for one chunk require approximately 6 GB. If all four chunks remain allocated concurrently, the total device footprint can approach 24 GB plus allocator state and temporary storage.

### Cooperative normalization examples

`advanced_cupy_cg.py` and `advanced_numba_cg.py` use 1,000,000,001 FP64 elements. Each full-sized array occupies slightly more than 8 GB. The input, output, and vectorized CuPy reference may therefore consume more than 24 GB of device memory, excluding allocator overhead and temporary operations.

### CUDA Graph example

`advanced_cupy_graph.py` uses persistent input, intermediate, and output arrays whose device addresses are captured by the graph. These allocations must remain valid until all graph launches have completed. Any reference expression used for validation can require additional temporary device storage, so validation should occur outside capture and after replay timing or profiling is complete.

### Stream examples

The stream examples use relatively small arrays but perform 10,000 transcendental-function iterations per element. They are compute-intensive by design so that stream scheduling is visible in the profiler.

### PyTorch inline CUDA extension

`advanced_torch_custom.py` uses small tensors and has a modest device-memory footprint. Its first execution can nevertheless require significant wall-clock time because `load_inline` generates build files, invokes the host compiler and `nvcc`, links a Python extension module, and loads the resulting shared library.

The extension cache must be writable. A persistent cache enables reuse across compatible jobs, while a job-specific cache prevents cross-job contention but forces recompilation. Simultaneous jobs that use the same extension name and cache directory can contend for build files or locks.

The current extension uses signed 32-bit `int` values for `numel()` and kernel indexing. The demonstration tensor contains only 1,000 elements, so this is safe. A reusable implementation intended for very large tensors should reject element counts greater than `INT_MAX` or use a suitable 64-bit index type.

The source validates the forward tensor as CUDA FP32 and makes both the saved input and incoming gradient contiguous. A production-oriented extension should additionally validate backward dtype, device, shape, element count, and CUDA launch status inside the extension boundary.

### Profiling overhead

Nsight Systems and especially the Nsight Compute `full` metric set add overhead. Profiled runtime should not be interpreted as uninstrumented application runtime.

## Correctness and synchronization notes

- CUDA kernel launches and asynchronous copies return before device work necessarily completes.
- Operations in one CUDA stream execute in submission order.
- Cross-stream dependencies require legacy default-stream semantics, an event, or explicit synchronization.
- Host access to a result requires completion of the operation that produces it.
- Pinned host memory enables asynchronous DMA transfers but does not guarantee overlap by itself.
- Overlap also depends on stream choice, dependency structure, copy-engine availability, kernel resource usage, and transfer direction.
- Device allocations and external stream wrappers must remain valid until all queued work using them has completed.
- PyTorch C++/CUDA extensions should launch kernels in PyTorch's current CUDA stream rather than assuming the legacy default stream.
- `advanced_torch_custom.py` makes the input and incoming gradient contiguous before passing their data pointers to kernels that use linear indexing.
- JIT extension compilation is separate from CUDA kernel execution and can dominate first-run wall-clock time.
- `TORCH_CUDA_ARCH_LIST` controls the GPU architectures included in the compiled PyTorch extension. The configured architecture must be compatible with the execution GPU.
- A cached PyTorch extension must remain compatible with the active source, extension name, Python, PyTorch, CUDA, compiler, and target-architecture configuration.
- Cooperative kernels require every block in the launch to be concurrently resident.
- CUDA Graph capture requires a capture-compatible operation sequence. Keep output allocation, synchronous host transfers, scalar extraction, and validation outside the capture region.
- A captured graph continues to reference the original device allocations. Rebinding a Python variable does not update the graph's captured addresses.
- FP64 global sums are order-dependent because parallel reductions and atomic additions do not reproduce NumPy's exact summation order.
- Warm-up launches separate JIT compilation and initialization effects from the principal profiled regions where the source implements them.
- Floating-point results should be checked with suitable relative and absolute tolerances rather than exact equality.

## Cleaning generated files

Remove logs and reports for one job with:

```bash
rm -f logs/run_advanced_<jobid>.out logs/run_advanced_<jobid>.err
rm -rf profiles_<jobid>
```

Remove all generated advanced-example logs and profile directories with:

```bash
rm -f logs/run_advanced_*.out logs/run_advanced_*.err
rm -rf profiles_[0-9]*
```

If a project-local PyTorch extension cache is configured and must be rebuilt, remove it separately:

```bash
rm -rf "${BASE_DIR}/.torch_extensions"
```

Review wildcard expansion carefully before removing files from a shared or important working directory.

## Scope

These programs are focused demonstrations of advanced CUDA Python and PyTorch extension mechanisms. They are intended for code study, correctness experiments, Nsight timeline analysis, and comparison of equivalent or related CuPy, Numba CUDA, and PyTorch techniques. They are not complete production benchmarks and do not attempt comprehensive parameter tuning, repeated statistical timing, exhaustive error handling, or portability across every CUDA software stack.

