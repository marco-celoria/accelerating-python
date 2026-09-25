#!/usr/bin/env python3

import argparse
from contextlib import contextmanager
from time import perf_counter

import cupy as cp
import numpy as np


# ============================================================
# DEFAULTS
# ============================================================

DEFAULT_TPB_X = 32
DEFAULT_TPB_Y = 32


# ============================================================
# CUDA C++ SOURCE
# ============================================================

CUDA_SOURCE = r'''
template <typename T>
__global__ void mm_kernel_base(
    const size_t A_num_rows,
    const size_t B_num_cols,
    const size_t A_num_cols,
    const T* __restrict__ A,
    const T* __restrict__ B,
    T* __restrict__ C
) {
    const size_t col = static_cast<size_t>(blockIdx.x) * 
                       static_cast<size_t>(blockDim.x) + 
                       static_cast<size_t>(threadIdx.x);

    const size_t row = static_cast<size_t>(blockIdx.y) * 
                       static_cast<size_t>(blockDim.y) + 
                       static_cast<size_t>(threadIdx.y);

    if (row >= A_num_rows || col >= B_num_cols) {
        return;
    }

    T acc = static_cast<T>(0);
    for (size_t k = 0; k < A_num_cols; ++k) {
        acc += A[row * A_num_cols + k] * B[k * B_num_cols + col];
    }

    C[row * B_num_cols + col] = acc;
}
'''


# ============================================================
# OPTIONAL NVTX SUPPORT
# ============================================================


@contextmanager
def nvtx_range(name: str, id_color: int = -1):
    """
    NVTX helper.

    If CuPy NVTX support is unavailable, this silently becomes a no-op.
    """
    pushed = False
    try:
        cp.cuda.nvtx.RangePush(name, id_color)
        pushed = True
    except Exception:
        pushed = False

    try:
        yield
    finally:
        if pushed:
            try:
                cp.cuda.nvtx.RangePop()
            except Exception:
                pass



# ============================================================
# VALIDATION / HELPERS
# ============================================================

def validate_positive_int(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0")


def validate_block_size(tpb_x: int, tpb_y: int) -> None:
    """
    Validate a rectangular two-dimensional CUDA thread block.

    The block is interpreted as:

        block = (tpb_x, tpb_y)

    where:
        x maps to output columns
        y maps to output rows
    """
    validate_positive_int("tpb_x", tpb_x)
    validate_positive_int("tpb_y", tpb_y)

    device = cp.cuda.Device()
    attributes = device.attributes

    threads_per_block = tpb_x * tpb_y

    max_threads_per_block = attributes["MaxThreadsPerBlock"]
    max_block_dim_x = attributes["MaxBlockDimX"]
    max_block_dim_y = attributes["MaxBlockDimY"]

    if threads_per_block > max_threads_per_block:
        raise ValueError(
            f"CUDA block ({tpb_x}, {tpb_y}) has "
            f"{threads_per_block} threads, but device "
            f"{device.id} supports at most "
            f"{max_threads_per_block} threads per block"
        )

    if tpb_x > max_block_dim_x:
        raise ValueError(
            f"tpb_x={tpb_x} exceeds "
            f"MaxBlockDimX={max_block_dim_x}"
        )

    if tpb_y > max_block_dim_y:
        raise ValueError(
            f"tpb_y={tpb_y} exceeds "
            f"MaxBlockDimY={max_block_dim_y}"
        )

def validate_matrix_shapes(a, b, c) -> None:
    if a.ndim != 2 or b.ndim != 2 or c.ndim != 2:
        raise ValueError("A, B, and C must all be 2D matrices")

    if a.shape[1] != b.shape[0]:
        raise ValueError(
            f"Incompatible matrix shapes: A{a.shape} and B{b.shape}; "
            "A.shape[1] must equal B.shape[0]"
        )

    expected_c_shape = (a.shape[0], b.shape[1])

    if c.shape != expected_c_shape:
        raise ValueError(
            f"Invalid C shape: got C{c.shape}, expected C{expected_c_shape}"
        )
    if a.dtype != b.dtype or a.dtype != c.dtype:
        raise TypeError(
            f"A, B, and C must have the same dtype; "
            f"got A={a.dtype}, B={b.dtype}, C={c.dtype}"
        )
    
    if not a.flags.c_contiguous:
        raise ValueError("A must be C-contiguous")
    if not b.flags.c_contiguous:
        raise ValueError("B must be C-contiguous")
    if not c.flags.c_contiguous:
        raise ValueError("C must be C-contiguous")


def compute_launch_geometry(
    a: cp.ndarray,
    b: cp.ndarray,
    tpb_x: int,
    tpb_y: int,
):
    """
    Compute a two-dimensional rectangular CUDA launch geometry.

    Thread mapping:

        threadIdx.x -> output column
        threadIdx.y -> output row

    Therefore:

        block = (tpb_x, tpb_y)
        grid  = (ceil(columns / tpb_x),
                 ceil(rows / tpb_y))
    """
    rows_c = a.shape[0]
    cols_c = b.shape[1]

    blocks_x = (cols_c + tpb_x - 1) // tpb_x
    blocks_y = (rows_c + tpb_y - 1) // tpb_y

    grid = (blocks_x, blocks_y)
    block = (tpb_x, tpb_y)

    return grid, block


def flop_count_matmul(n: int, l: int, m: int) -> float:
    """
    Return the conventional GEMM count, 2*n*l*m FLOPs.

    Each multiply-add is counted as two floating-point operations.
    """
    return 2.0 * float(n) * float(l) * float(m)



def dtype_tolerances(dtype: np.dtype):
    """
    Return pragmatic validation tolerances for this benchmark.

    These values are not formal forward-error bounds. Appropriate
    tolerances can depend on the inner dimension, input scaling,
    accumulation order, implementation, and GPU math mode.
    """
    dtype = np.dtype(dtype)

    if dtype == np.dtype(np.float32):
        return 1.0e-4, 1.0e-4

    if dtype == np.dtype(np.float64):
        return 1.0e-10, 1.0e-10

    raise TypeError("dtype must be either float32 or float64")


def compile_kernels():
    """
    Compile templated CUDA kernel specializations.
    """
    name_expressions = [
        "mm_kernel_base<float>",
        "mm_kernel_base<double>",
    ]

    module = cp.RawModule(
        code=CUDA_SOURCE,
        options=("-std=c++17", "--generate-line-info"),
        name_expressions=name_expressions,
    )

    ker_float = module.get_function("mm_kernel_base<float>")
    ker_double = module.get_function("mm_kernel_base<double>")

    return ker_float, ker_double


def gpu_kernel_time_seconds(
    kernel,
    grid,
    block,
    a_num_rows,
    b_num_cols,
    a_num_cols,
    a_d: cp.ndarray,
    b_d: cp.ndarray,
    c_d: cp.ndarray,
    start_event,
    stop_event,
) -> float:
    """
    Time one CUDA kernel launch using CUDA events.

    CuPy returns elapsed event time in milliseconds.
    """

    start_event.record()

    with nvtx_range("Launch CuPy RawKernel matmul"):
        kernel(
            grid,
            block,
            (
                a_num_rows,
                b_num_cols,
                a_num_cols,
                a_d,
                b_d,
                c_d,
            ),
        )

    stop_event.record()
    stop_event.synchronize()

    return cp.cuda.get_elapsed_time(start_event, stop_event) * 1.0e-3


def validate_selected_dtype(
    a: cp.ndarray,
    b: cp.ndarray,
    c: cp.ndarray,
    dtype: np.dtype,
) -> None:
    expected_dtype = np.dtype(dtype)

    if (
        np.dtype(a.dtype) != expected_dtype
        or np.dtype(b.dtype) != expected_dtype
        or np.dtype(c.dtype) != expected_dtype
    ):
        raise TypeError(
            "A, B, and C must match the selected dtype; "
            f"expected {expected_dtype}, got "
            f"A={a.dtype}, B={b.dtype}, C={c.dtype}"
        )


def validate_input_matrices(a, b, dtype: np.dtype) -> None:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("A and B must both be 2D matrices")

    if a.shape[1] != b.shape[0]:
        raise ValueError(
            f"Incompatible matrix shapes: A{a.shape} and B{b.shape}; "
            "A.shape[1] must equal B.shape[0]"
        )

    expected_dtype = np.dtype(dtype)

    if np.dtype(a.dtype) != expected_dtype:
        raise TypeError(
            f"A must have dtype {expected_dtype}, got {a.dtype}"
        )

    if np.dtype(b.dtype) != expected_dtype:
        raise TypeError(
            f"B must have dtype {expected_dtype}, got {b.dtype}"
        )

    if not a.flags.c_contiguous:
        raise ValueError("A must be C-contiguous")

    if not b.flags.c_contiguous:
        raise ValueError("B must be C-contiguous")


# ============================================================
# BENCHMARK DRIVER
# ============================================================

def matmul(
    a_h: np.ndarray,
    b_h: np.ndarray,
    iters: int,
    kernel,
    tpb_x: int,
    tpb_y: int,
    dtype: np.dtype,
) -> bool:
    
    validate_positive_int("iters", iters)
    validate_input_matrices(a_h, b_h, dtype)
    # Preallocate CPU output to avoid allocation in the timed CPU loop.
    check_h = np.empty((a_h.shape[0], b_h.shape[1]), dtype=dtype)
    validate_matrix_shapes(a_h, b_h, check_h)
    with nvtx_range("H2D input copies"):
        a_d = cp.asarray(a_h)
        b_d = cp.asarray(b_h)
    with nvtx_range("Device output allocation"):
        c_d = cp.empty((a_d.shape[0], b_d.shape[1]), dtype=a_d.dtype,)

    a_num_rows = np.uint64(a_d.shape[0])
    b_num_cols = np.uint64(b_d.shape[1])
    a_num_cols = np.uint64(a_d.shape[1])

    validate_block_size(tpb_x, tpb_y)
    validate_matrix_shapes(a_d, b_d, c_d)
    validate_selected_dtype(a_d, b_d, c_d, dtype)
    grid, block = compute_launch_geometry(a_d, b_d ,tpb_x, tpb_y)

    print(f"blockspergrid   = {grid}")
    print(f"threadsperblock = {block}")

    # --------------------------------------------------------
    # Warm-up launch
    #
    # The warm-up excludes compilation, module loading, and
    # first-launch setup from the measured iterations
    # --------------------------------------------------------
    print("Warming up CUDA kernel...")

    with nvtx_range("Warmup CuPy RawKernel matmul"):
        kernel(
            grid,
            block,
            (
                a_num_rows,
                b_num_cols,
                a_num_cols,
                a_d,
                b_d,
                c_d,
            ),
        )

    cp.cuda.Device().synchronize()
    
    print("Warming up CPU NumPy matmul...")

    with nvtx_range("CPU NumPy warmup", id_color=4):
        np.matmul(a_h, b_h, out=check_h)

    print("Warmup complete.")
    print("Timings below exclude RawModule compilation, first-launch overhead, and the initial NumPy matmul call.")

    timings_gpu = np.empty(iters, dtype=np.float64)
    timings_cpu = np.empty(iters, dtype=np.float64)

    rtol, atol = dtype_tolerances(dtype)
    start_event = cp.cuda.Event()
    stop_event = cp.cuda.Event()
    total_start_event = cp.cuda.Event()
    total_stop_event = cp.cuda.Event()

    total_start_event.record()
    for i in range(iters):
        # ----------------------------------------------------
        # GPU timing with CUDA events
        # ----------------------------------------------------
        timings_gpu[i] = gpu_kernel_time_seconds(
            kernel=kernel,
            grid=grid,
            block=block,
            a_num_rows=a_num_rows,
            b_num_cols=b_num_cols,
            a_num_cols=a_num_cols,
            a_d=a_d,
            b_d=b_d,
            c_d=c_d,
            start_event=start_event,
            stop_event=stop_event,
        )

    total_stop_event.record()
    total_stop_event.synchronize()
    gpu_total_time = cp.cuda.get_elapsed_time(total_start_event, total_stop_event) * 1.0e-3
    print(f"GPU timing summary: total={gpu_total_time:.6e} s, sum_of_iterations={timings_gpu.sum():.6e} s")

    # ----------------------------------------------------
    # CPU timing with preallocated output
    # ----------------------------------------------------
    cpu_total_start = perf_counter()
    for i in range(iters):
        with nvtx_range("CPU NumPy matmul"):
            cpu_t0 = perf_counter()
            np.matmul(a_h, b_h, out=check_h)
            cpu_t1 = perf_counter()

        timings_cpu[i] = cpu_t1 - cpu_t0

    cpu_total_time = perf_counter() - cpu_total_start
    print(f"CPU timing summary: total={cpu_total_time:.6e} s, sum_of_iterations={timings_cpu.sum():.6e} s")

    for i in range(iters):
        print(
            f"Iteration {i + 1}/{iters}: "
            f"gpu_time={timings_gpu[i]:.6e} s, "
            f"cpu_time={timings_cpu[i]:.6e} s"
        )

    # ----------------------------------------------------
    # Correctness check
    # ----------------------------------------------------
    with nvtx_range("D2H result copy"):
        c_h = cp.asnumpy(c_d)
    
    gpu_finite = bool(np.isfinite(c_h).all())
    cpu_finite = bool(np.isfinite(check_h).all())
    finite = gpu_finite and cpu_finite

    if finite:
        diff = c_h - check_h
        diff_norm = float(np.linalg.norm(diff))
        ref_norm = float(np.linalg.norm(check_h))
        err_max = float(np.max(np.abs(diff)))

        relative_error = (
            diff_norm / ref_norm
            if ref_norm != 0.0
            else diff_norm
        )

        check_passed = bool(
            np.allclose(
                c_h,
                check_h,
                rtol=rtol,
                atol=atol,
            )
        )
    else:
        diff_norm = float("nan")
        ref_norm = float("nan")
        err_max = float("nan")
        relative_error = float("nan")
        check_passed = False

    print()
    print("Correctness validation")
    print(f"GPU result finite:   {gpu_finite}")
    print(f"CPU result finite:   {cpu_finite}")
    print(f"Both results finite: {finite}")
    print(f"Check passed:        {check_passed}")
    print(f"Error norm:          {diff_norm:.6e}")
    print(f"Maximum abs. error:  {err_max:.6e}")
    print(f"Relative error:      {relative_error:.6e}")
    print(f"rtol:                {rtol:.6e}")
    print(f"atol:                {atol:.6e}")

    gpu_mean = float(np.mean(timings_gpu))
    gpu_median = float(np.median(timings_gpu))
    gpu_min = float(np.min(timings_gpu))
    gpu_std = (float(np.std(timings_gpu, ddof=1)) if iters > 1 else 0.0)

    cpu_mean = float(np.mean(timings_cpu))
    cpu_median = float(np.median(timings_cpu))
    cpu_min = float(np.min(timings_cpu))
    cpu_std = (float(np.std(timings_cpu, ddof=1)) if iters > 1 else 0.0)

    n = a_d.shape[0]
    l = a_d.shape[1]
    m = b_d.shape[1]

    flops = flop_count_matmul(n, l, m)
    gflops = flops * 1.0e-9
    
    if gpu_median <= 0.0:
        raise RuntimeError("GPU median time is not positive; increase the workload or time multiple launches per sample.")
    if cpu_median <= 0.0:
        raise RuntimeError("CPU median time is not positive.")

    gpu_gflops_s = gflops / gpu_median
    cpu_gflops_s = gflops / cpu_median
    speedup = cpu_median / gpu_median

    print()
    print("============================================================")
    print("Benchmark summary")
    print("============================================================")
    print("Implementation:               CuPy RawModule naive CUDA kernel")
    print("Timing scope:                 compute only")
    print("GPU timing method:            CUDA events")
    print("CPU timing method:            perf_counter")
    print("Transfers included:           no")
    print("Compilation included:         no")
    print("CPU comparison:               NumPy matmul; BLAS and thread dependent")
    print(f"Matrix shapes:                A={a_d.shape}, B={b_d.shape}, C={c_d.shape}")
    print(f"dtype:                        {np.dtype(dtype)}")
    print(f"iterations:                   {iters}")
    print(f"Conventional GEMM FLOPs:      {flops:.0f}")
    print(f"GPU elapsed mean:             {gpu_mean:.6e} s")
    print(f"GPU elapsed median:           {gpu_median:.6e} s")
    print(f"GPU elapsed minimum:          {gpu_min:.6e} s")
    print(f"GPU elapsed stddev:           {gpu_std:.6e} s")
    print(f"CPU elapsed mean:             {cpu_mean:.6e} s")
    print(f"CPU elapsed median:           {cpu_median:.6e} s")
    print(f"CPU elapsed minimum:          {cpu_min:.6e} s")
    print(f"CPU elapsed stddev:           {cpu_std:.6e} s")
    print(f"NumPy/GPU compute-time ratio: {speedup:.2f}x")
    print(f"GPU GFLOP/s median:           {gpu_gflops_s:.4f}")
    print(f"CPU GFLOP/s median:           {cpu_gflops_s:.4f}")
    print(f"Check passed:                 {check_passed}")
    print("Correctness is checked against NumPy using the same input dtype.")
    return check_passed


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "CuPy RawModule baseline matrix multiplication benchmark "
            "using a custom CUDA C++ kernel"
        )
    )

    parser.add_argument(
        "n",
        nargs="?",
        type=int,
        default=1000,
        help="Rows of A and C",
    )

    parser.add_argument(
        "m",
        nargs="?",
        type=int,
        default=1000,
        help="Columns of B and C",
    )

    parser.add_argument(
        "l",
        nargs="?",
        type=int,
        default=1000,
        help="Columns of A / rows of B",
    )

    parser.add_argument(
        "iters",
        nargs="?",
        type=int,
        default=10,
        help="Number of benchmark iterations",
    )

    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float64",
        help=("Floating-point precision. FP64 throughput is strongly GPU-dependent and may be much lower on non-HPC GPUs."),
    )


    parser.add_argument(
        "--tpb-x",
        type=int,
        default=DEFAULT_TPB_X,
        help=("CUDA threads per block in the x direction. The x dimension maps to output columns."),
    )

    parser.add_argument(
        "--tpb-y",
        type=int,
        default=DEFAULT_TPB_Y,
        help=(
        "CUDA threads per block in the y direction. The y dimension maps to output rows."),
    )


    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed",
    )

    args = parser.parse_args()

    validate_positive_int("n", args.n)
    validate_positive_int("m", args.m)
    validate_positive_int("l", args.l)
    validate_positive_int("iters", args.iters)
    if cp.cuda.runtime.getDeviceCount() <= 0:
        raise RuntimeError("No CUDA device found")
    validate_block_size(args.tpb_x, args.tpb_y)

    if args.dtype == "float32":
        dtype_np = np.dtype(np.float32)
    elif args.dtype == "float64":
        dtype_np = np.dtype(np.float64)
    else:
        raise TypeError("dtype must be either float32 or float64")

    n = args.n
    m = args.m
    l = args.l

    print(f"A shape: {n} x {l}")
    print(f"B shape: {l} x {m}")
    print(f"C shape: {n} x {m}")
    print(f"dtype:   {dtype_np}")
    print(f"iters:   {args.iters}")
    print(f"tpb_x:   {args.tpb_x}")
    print(f"tpb_y:   {args.tpb_y}")
    print(f"block:   ({args.tpb_x}, {args.tpb_y})")
    
    rng = np.random.default_rng(args.seed)

    x_h = rng.standard_normal((n, l)).astype(dtype_np)
    y_h = rng.standard_normal((l, m)).astype(dtype_np)

    print("Compiling RawModule kernels...")
    ker_float, ker_double = compile_kernels()
    print("Compilation complete.")

    if dtype_np == np.dtype(np.float32):
        kernel = ker_float
    elif dtype_np == np.dtype(np.float64):
        kernel = ker_double
    else:
        raise TypeError("dtype must be either float32 or float64")

    check_passed = matmul(
        a_h=x_h,
        b_h=y_h,
        iters=args.iters,
        kernel=kernel,
        tpb_x=args.tpb_x,
        tpb_y=args.tpb_y,
        dtype=dtype_np,
    )

    return 0 if check_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

