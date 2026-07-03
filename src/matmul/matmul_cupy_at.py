#!/usr/bin/env python3

import argparse
import time
from contextlib import contextmanager

import cupy as cp
import numpy as np


# ============================================================
# OPTIONAL NVTX SUPPORT
# ============================================================

@contextmanager
def nvtx_range(name: str):
    """
    NVTX helper.

    If CuPy NVTX support is unavailable, this silently becomes a no-op.
    """
    pushed = False

    try:
        cp.cuda.nvtx.RangePush(name)
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


def flop_count_matmul(n: int, l: int, m: int) -> float:
    """
    FLOP count for matrix multiplication:

        C[n,m] = A[n,l] @ B[l,m]

    Per output element:
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
        return 1.0e-12, 1.0e-12

    raise TypeError("dtype must be either float32 or float64")


def gpu_matmul_time_seconds(
    a: cp.ndarray,
    b: cp.ndarray,
    c: cp.ndarray,
) -> float:
    """
    Time one CuPy matmul using CUDA events.

    This measures GPU elapsed time for:

        cp.matmul(a, b, out=c)

    CuPy event elapsed time is returned in milliseconds, so we convert to seconds.
    """
    start_event = cp.cuda.Event()
    stop_event = cp.cuda.Event()

    cp.cuda.Device().synchronize()

    start_event.record()

    with nvtx_range("GPU CuPy matmul"):
        cp.matmul(a, b, out=c)

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
    dtype: np.dtype,
) -> None:
    validate_positive_int("iters", iters)
    validate_matrix_shapes(a, b, c)

    # Copy inputs once for CPU reference.
    with nvtx_range("D2H input copies"):
        a_h = cp.asnumpy(a)
        b_h = cp.asnumpy(b)

    # Preallocate CPU output to avoid allocation inside timed loop.
    check_h = np.empty((a_h.shape[0], b_h.shape[1]), dtype=dtype)

    # --------------------------------------------------------
    # Warm-up.
    #
    # This excludes first-call overhead from the measured timings.
    # CuPy will typically dispatch to cuBLAS for matmul.
    # --------------------------------------------------------
    print("Warming up CuPy matmul...")

    with nvtx_range("Warmup CuPy matmul"):
        cp.matmul(a, b, out=c)

    cp.cuda.Device().synchronize()

    print("Warmup complete. Timings below exclude first-call overhead.")

    timings_gpu = np.empty(iters, dtype=np.float64)
    timings_cpu = np.empty(iters, dtype=np.float64)

    rtol, atol = dtype_tolerances(dtype)

    all_checks_passed = True

    for i in range(iters):
        # ----------------------------------------------------
        # GPU timing with CUDA events
        # ----------------------------------------------------
        timings_gpu[i] = gpu_matmul_time_seconds(a, b, c)

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
        description="CuPy built-in matrix multiplication benchmark"
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

    cp.random.seed(args.seed)

    with nvtx_range("GPU allocation and initialization"):
        a_d = cp.random.standard_normal((n, l), dtype=dtype_cp)
        b_d = cp.random.standard_normal((l, m), dtype=dtype_cp)
        c_d = cp.empty((n, m), dtype=dtype_cp)

    matmul(
        a=a_d,
        b=b_d,
        c=c_d,
        iters=args.iters,
        dtype=dtype_np,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

