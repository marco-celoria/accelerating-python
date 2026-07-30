#!/usr/bin/env python3

import argparse
import math
import time
from contextlib import contextmanager

import cupy as cp
import numpy as np


# ============================================================
# DEFAULTS
# ============================================================

DEFAULT_TPB = 16


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
    const size_t col = static_cast<size_t>(blockIdx.x) * static_cast<int>(blockDim.x)
                  + static_cast<size_t>(threadIdx.x);

    const size_t row = static_cast<size_t>(blockIdx.y) * static_cast<int>(blockDim.y)
                  + static_cast<size_t>(threadIdx.y);

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


def validate_tpb(tpb: int) -> None:
    validate_positive_int("tpb", tpb)

    threads_per_block = tpb * tpb

    if threads_per_block > 1024:
        raise ValueError(
            f"tpb={tpb} gives {threads_per_block} threads/block, "
            "but CUDA allows at most 1024"
        )


def validate_matrix_shapes(a: cp.ndarray, b: cp.ndarray, c: cp.ndarray) -> None:
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


def compute_launch_geometry(a: cp.ndarray, b: cp.ndarray, tpb: int):
    rows_c = a.shape[0]
    cols_c = b.shape[1]

    blocks_x = math.ceil(cols_c / tpb)
    blocks_y = math.ceil(rows_c / tpb)

    grid = (blocks_x, blocks_y)
    block = (tpb, tpb)

    return grid, block


def flop_count_matmul(n: int, l: int, m: int) -> float:
    """
    FLOP count for naive matrix multiplication:

        C[n,m] = A[n,l] @ B[l,m]

    Per C element:
        l multiplications + (l - 1) additions = 2*l - 1 FLOPs

    Total:
        n * m * (2*l - 1)
    """
    return float(n) * float(m) * float(2 * l - 1)

def dtype_tolerances(dtype: np.dtype):
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
    a: cp.ndarray,
    b: cp.ndarray,
    c: cp.ndarray,
) -> float:
    """
    Time one CUDA kernel launch using CUDA events.

    CuPy returns elapsed event time in milliseconds.
    """
    start_event = cp.cuda.Event()
    stop_event = cp.cuda.Event()

    cp.cuda.Device().synchronize()

    start_event.record()

    with nvtx_range("GPU RawKernel matmul"):
        kernel(
            grid,
            block,
            (
                np.uint64(a.shape[0]),
                np.uint64(b.shape[1]),
                np.uint64(a.shape[1]),
                a,
                b,
                c,
            ),
        )

    stop_event.record()
    stop_event.synchronize()

    return cp.cuda.get_elapsed_time(start_event, stop_event) * 1.0e-3


# ============================================================
# BENCHMARK DRIVER
# ============================================================

def matmul(
    a: cp.ndarray,
    b: cp.ndarray,
    c: cp.ndarray,
    iters: int,
    kernel,
    tpb: int,
    dtype: np.dtype,
) -> None:
    validate_positive_int("iters", iters)
    validate_tpb(tpb)
    validate_matrix_shapes(a, b, c)

    grid, block = compute_launch_geometry(a, b, tpb)

    print(f"blockspergrid   = {grid}")
    print(f"threadsperblock = {block}")

    # Copy inputs once for CPU reference.
    with nvtx_range("D2H input copies"):
        a_h = cp.asnumpy(a)
        b_h = cp.asnumpy(b)

    # Preallocate CPU output to avoid allocation in the timed CPU loop.
    check_h = np.empty((a_h.shape[0], b_h.shape[1]), dtype=dtype)

    # --------------------------------------------------------
    # Warm-up launch
    #
    # This excludes first-launch overhead from timings.
    # RawModule compilation already happened at get_function(), but the first
    # actual launch can still include setup overhead.
    # --------------------------------------------------------
    print("Warming up CUDA kernel...")

    with nvtx_range("Warmup RawKernel matmul"):
        kernel(
            grid,
            block,
            (
                np.uint64(a.shape[0]),
                np.uint64(b.shape[1]),
                np.uint64(a.shape[1]),
                a,
                b,
                c,
            ),
        )

    cp.cuda.Device().synchronize()

    print("Warmup complete. Timings below exclude first-launch overhead.")

    timings_gpu = np.empty(iters, dtype=np.float64)
    timings_cpu = np.empty(iters, dtype=np.float64)

    rtol, atol = dtype_tolerances(dtype)

    all_checks_passed = True

    for i in range(iters):
        # ----------------------------------------------------
        # GPU timing with CUDA events
        # ----------------------------------------------------
        timings_gpu[i] = gpu_kernel_time_seconds(
            kernel=kernel,
            grid=grid,
            block=block,
            a=a,
            b=b,
            c=c,
        )

        # ----------------------------------------------------
        # CPU timing with preallocated output
        # ----------------------------------------------------
        cpu_t0 = time.perf_counter()

        with nvtx_range("CPU NumPy matmul"):
            np.matmul(a_h, b_h, out=check_h)

        cpu_t1 = time.perf_counter()

        timings_cpu[i] = cpu_t1 - cpu_t0

        # ----------------------------------------------------
        # Correctness check
        # ----------------------------------------------------
        with nvtx_range("D2H result copy"):
            c_h = cp.asnumpy(c)

        ok = np.allclose(c_h, check_h, rtol=rtol, atol=atol)
        all_checks_passed = all_checks_passed and ok

        err_norm = np.linalg.norm(c_h - check_h)
        err_max = np.max(np.abs(c_h - check_h))

        print(
            f"Iteration {i + 1}/{iters}: "
            f"check={ok}, "
            f"norm_error={err_norm:.6e}, "
            f"max_abs_error={err_max:.6e}, "
            f"gpu_time={timings_gpu[i]:.6e} s, "
            f"cpu_time={timings_cpu[i]:.6e} s"
        )

    gpu_mean = float(timings_gpu.mean())
    gpu_std = float(timings_gpu.std())
    cpu_mean = float(timings_cpu.mean())
    cpu_std = float(timings_cpu.std())

    n = a.shape[0]
    l = a.shape[1]
    m = b.shape[1]

    flops = flop_count_matmul(n, l, m)
    gflops = flops * 1.0e-9

    gpu_gflops_s = gflops / gpu_mean
    cpu_gflops_s = gflops / cpu_mean

    print()
    print("============================================================")
    print("Benchmark summary")
    print("============================================================")
    print(f"Matrix shapes:       A={a.shape}, B={b.shape}, C={c.shape}")
    print(f"dtype:               {np.dtype(dtype)}")
    print(f"iterations:          {iters}")
    print(f"tpb:                 {tpb}")
    print(f"FLOPs per matmul:    {flops:.0f}")
    print(f"All checks passed:   {all_checks_passed}")
    print(f"GPU elapsed mean:    {gpu_mean:.6e} s")
    print(f"GPU elapsed std:     {gpu_std:.6e} s")
    print(f"CPU elapsed mean:    {cpu_mean:.6e} s")
    print(f"CPU elapsed std:     {cpu_std:.6e} s")
    print(f"Speedup:             {cpu_mean / gpu_mean:.2f}x")
    print(f"GPU GFLOP/s:         {gpu_gflops_s:.4f}")
    print(f"CPU GFLOP/s:         {cpu_gflops_s:.4f}")


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
        default=99,
        help="Rows of A and C",
    )

    parser.add_argument(
        "m",
        nargs="?",
        type=int,
        default=123,
        help="Columns of B and C",
    )

    parser.add_argument(
        "l",
        nargs="?",
        type=int,
        default=78,
        help="Columns of A / rows of B",
    )

    parser.add_argument(
        "iters",
        nargs="?",
        type=int,
        default=1,
        help="Number of benchmark iterations",
    )

    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float64",
        help="Floating-point precision",
    )

    parser.add_argument(
        "--tpb",
        type=int,
        default=DEFAULT_TPB,
        help="Threads per block dimension. Block shape is tpb x tpb.",
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
    validate_tpb(args.tpb)

    if cp.cuda.runtime.getDeviceCount() <= 0:
        raise RuntimeError("No CUDA device found")

    if args.dtype == "float32":
        dtype_cp = cp.float32
        dtype_np = np.dtype(np.float32)
    elif args.dtype == "float64":
        dtype_cp = cp.float64
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
    print(f"tpb:     {args.tpb}")

    cp.random.seed(args.seed)

    with nvtx_range("GPU allocation and initialization"):
        x_d = cp.random.standard_normal((n, l), dtype=dtype_cp)
        y_d = cp.random.standard_normal((l, m), dtype=dtype_cp)
        z_d = cp.zeros((n, m), dtype=dtype_cp)

    print("Compiling RawModule kernels...")
    ker_float, ker_double = compile_kernels()
    print("Compilation complete.")

    if dtype_np == np.dtype(np.float32):
        kernel = ker_float
    elif dtype_np == np.dtype(np.float64):
        kernel = ker_double
    else:
        raise TypeError("dtype must be either float32 or float64")

    matmul(
        a=x_d,
        b=y_d,
        c=z_d,
        iters=args.iters,
        kernel=kernel,
        tpb=args.tpb,
        dtype=dtype_np,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

