#!/usr/bin/env python3

import argparse
from contextlib import contextmanager
from time import perf_counter

import numpy as np
from numba import cuda, float32, float64

try:
    import cupy as cp
except ImportError:
    cp = None

# ============================================================
# OPTIONAL NVTX SUPPORT
# ============================================================

@contextmanager
def nvtx_range(name: str, id_color: int = -1):
    """
    NVTX helper.

    If CuPy or CuPy NVTX support is unavailable, this becomes a no-op.
    """
    pushed = False

    if cp is not None:
        try:
            cp.cuda.nvtx.RangePush(name, id_color)
            pushed = True
        except Exception:
            pass

    try:
        yield
    finally:
        if pushed:
            try:
                cp.cuda.nvtx.RangePop()
            except Exception:
                pass


# Compile-time square tile dimension.
#
# The kernel must be launched with exactly (TPB, TPB) threads per block
# because the shared-memory arrays and matrix indexing use TPB directly.
# Controls threads per block and shared memory usage.
# The computation will be done on blocks of TPBxTPB elements.
# TPB * TPB <= device maximum threads per block
TPB = 16


# ============================================================
# VALIDATION / HELPERS
# ============================================================

def validate_positive_int(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0")


def validate_block_size(tpb_x: int, tpb_y: int) -> None:
    validate_positive_int("tpb_x", tpb_x)
    validate_positive_int("tpb_y", tpb_y)

    device = cuda.get_current_device()
    threads = tpb_x * tpb_y

    if threads > device.MAX_THREADS_PER_BLOCK:
        raise ValueError(
            f"CUDA block has {threads} threads, but device "
            f"{device.name} supports at most "
            f"{device.MAX_THREADS_PER_BLOCK} threads per block"
        )

    if tpb_x > device.MAX_BLOCK_DIM_X:
        raise ValueError(
            f"tpb_x={tpb_x} exceeds MAX_BLOCK_DIM_X="
            f"{device.MAX_BLOCK_DIM_X}"
        )

    if tpb_y > device.MAX_BLOCK_DIM_Y:
        raise ValueError(
            f"tpb_y={tpb_y} exceeds MAX_BLOCK_DIM_Y="
            f"{device.MAX_BLOCK_DIM_Y}"
        )


def validate_matrix_shapes(a, b, c) -> None:
    if len(a.shape) != 2 or len(b.shape) != 2 or len(c.shape) != 2:
        raise ValueError("A, B, and C must all be 2D matrices")

    if a.shape[1] != b.shape[0]:
        raise ValueError(
            f"Incompatible matrix shapes: A{a.shape} and B{b.shape}; "
            f"A.shape[1] must equal B.shape[0]"
        )

    expected_c_shape = (a.shape[0], b.shape[1])

    if c.shape != expected_c_shape:
        raise ValueError(
            f"Invalid C shape: got C{c.shape}, expected C{expected_c_shape}"
        )


@cuda.jit
def matmul_shared_float64(A, B, C):
    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    s_A = cuda.shared.array(shape=(TPB, TPB), dtype=float64)
    s_B = cuda.shared.array(shape=(TPB, TPB), dtype=float64)
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    col = bx * TPB + tx
    row = by * TPB + ty
    
    # The dot product is chunked into dot products of TPB-long vectors.
    acc = float64(0.0)
    num_tiles = (A.shape[1] + TPB - 1) // TPB
    
    for tile in range(num_tiles):
        col_A = tile * TPB + tx
        row_B = tile * TPB + ty
        
        # Preload data into shared memory
        if row < A.shape[0] and col_A < A.shape[1]:
            s_A[ty, tx] = A[row, col_A]  # copy A to shared mem
        else:
            s_A[ty, tx] = float64(0.0)
            
        if row_B < B.shape[0] and col < B.shape[1]:
            s_B[ty, tx] = B[row_B, col]  # copy B to shared mem
        else:
            s_B[ty, tx] = float64(0.0)
            
        # Wait until all threads finish preloading
        cuda.syncthreads()
        
        # Computes partial product on the shared memory
        for k in range(TPB):
            acc += s_A[ty, k] * s_B[k, tx]
            
        # Wait until all threads finish computing
        cuda.syncthreads()
        
    if row < C.shape[0] and col < C.shape[1]:
        C[row, col] = acc

       
@cuda.jit
def matmul_shared_float32(A, B, C):
    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    s_A = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    s_B = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    col = bx * TPB + tx
    row = by * TPB + ty
    
    # The dot product is chunked into dot products of TPB-long vectors.
    acc = float32(0.0)
    num_tiles = (A.shape[1] + TPB - 1) // TPB
    
    for tile in range(num_tiles):
        col_A = tile * TPB + tx
        row_B = tile * TPB + ty
        
        # Preload data into shared memory
        if row < A.shape[0] and col_A < A.shape[1]:
            s_A[ty, tx] = A[row, col_A]  # copy A to shared mem
        else:
            s_A[ty, tx] = float32(0.0)
            
        if row_B < B.shape[0] and col < B.shape[1]:
            s_B[ty, tx] = B[row_B, col]  # copy B to shared mem
        else:
            s_B[ty, tx] = float32(0.0)
            
        # Wait until all threads finish preloading
        cuda.syncthreads()
        
        # Computes partial product on the shared memory
        for k in range(TPB):
            acc += s_A[ty, k] * s_B[k, tx]
            
        # Wait until all threads finish computing
        cuda.syncthreads()
        
    if row < C.shape[0] and col < C.shape[1]:
        C[row, col] = acc



def compute_launch_geometry(a, b):
    threadsperblock = (TPB, TPB)

    rows_c = a.shape[0]
    cols_c = b.shape[1]
    
    blockspergrid_x = (cols_c + TPB - 1) // TPB
    blockspergrid_y = (rows_c + TPB - 1) // TPB
    
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    return blockspergrid, threadsperblock


def flop_count_matmul(n: int, l: int, m: int) -> float:
    """
    Operation count for naive matrix multiplication:
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



def gpu_kernel_time_seconds(
    kernel,
    blockspergrid,
    threadsperblock,
    a_d,
    b_d,
    c_d,
    start_event,
    stop_event,
) -> float:
    """
    Time one shared-memory CUDA kernel launch using CUDA events.
    """
    start_event.record()
    
    with nvtx_range("Launch shared-memory matmul", id_color=1):
        kernel[blockspergrid, threadsperblock](a_d, b_d, c_d)

    stop_event.record()
    stop_event.synchronize()
    return cuda.event_elapsed_time(start_event, stop_event) * 1.0e-3


def validate_matrix_dtypes(a, b, c, dtype: np.dtype) -> None:
    expected_dtype = np.dtype(dtype)

    actual_dtypes = (
        np.dtype(a.dtype),
        np.dtype(b.dtype),
        np.dtype(c.dtype),
    )

    if actual_dtypes != (
        expected_dtype,
        expected_dtype,
        expected_dtype,
    ):
        raise TypeError(
            "A, B, and C must match the selected dtype; "
            f"expected {expected_dtype}, got "
            f"A={actual_dtypes[0]}, "
            f"B={actual_dtypes[1]}, "
            f"C={actual_dtypes[2]}"
        )


def matmul(
    a_d,
    b_d,
    c_d,
    iters: int,
    kernel,
    dtype: np.dtype,
) -> bool:
    validate_positive_int("iters", iters)
    validate_matrix_shapes(a_d, b_d, c_d)
    validate_matrix_dtypes(a_d, b_d, c_d, dtype)

    # TPB is a compile-time kernel constant, so the launch must use exactly TPB x TPB threads per block.
    validate_block_size(TPB, TPB)

    blockspergrid, threadsperblock = compute_launch_geometry(
        a_d,
        b_d,
    )
    
    shared_bytes = 2 * TPB * TPB * np.dtype(dtype).itemsize
    print(f"blockspergrid    = {blockspergrid}")
    print(f"threadsperblock  = {threadsperblock}")
    print(f"tile dimension   = {TPB} x {TPB}")
    print(f"shared memory    = {shared_bytes} bytes/block")

    # Copy inputs once for CPU reference.
    with nvtx_range("D2H input copies", id_color=2):
        a_h = a_d.copy_to_host()
        b_h = b_d.copy_to_host()

    # Preallocate CPU reference output.
    check_h = np.empty((a_h.shape[0], b_h.shape[1]), dtype=dtype)

    # --------------------------------------------------------
    # Warm-up launch
    # --------------------------------------------------------
    print("Warming up CUDA kernel / compiling Numba specialization...")

    with nvtx_range("Warmup kernel", id_color=3):
        kernel[blockspergrid, threadsperblock](a_d, b_d, c_d)

    cuda.synchronize()

    print("Warming up CPU NumPy matmul...")

    with nvtx_range("CPU NumPy warmup", id_color=4):
        np.matmul(a_h, b_h, out=check_h)

    print("Warmup complete.")
    print("Timings exclude CUDA JIT compilation and the initial NumPy matmul call.")

    timings_gpu = np.empty(iters, dtype=np.float64)
    timings_cpu = np.empty(iters, dtype=np.float64)

    rtol, atol = dtype_tolerances(dtype)
    start_event = cuda.event()
    stop_event = cuda.event()
    for i in range(iters):
        # ----------------------------------------------------
        # GPU timing using CUDA events
        # ----------------------------------------------------
        timings_gpu[i] = gpu_kernel_time_seconds(
            kernel=kernel,
            blockspergrid=blockspergrid,
            threadsperblock=threadsperblock,
            a_d=a_d,
            b_d=b_d,
            c_d=c_d,
            start_event=start_event,
            stop_event=stop_event,
        )


    for i in range(iters):
        with nvtx_range("CPU NumPy matmul", id_color=4):
            cpu_t0 = perf_counter()
            np.matmul(a_h, b_h, out=check_h)
            cpu_t1 = perf_counter()

        timings_cpu[i] = cpu_t1 - cpu_t0

    for i in range(iters):
        print(
            f"Iteration {i + 1}/{iters}: "
            f"gpu_time={timings_gpu[i]:.6e} s, "
            f"cpu_time={timings_cpu[i]:.6e} s"
        )

    # --------------------------------------------------------
    # Correctness check
    # --------------------------------------------------------
    with nvtx_range("D2H result copy", id_color=5):
        c_h = c_d.copy_to_host()

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

    gpu_gflops_s = gflops / gpu_median
    cpu_gflops_s = gflops / cpu_median
    speedup = cpu_median / gpu_median

    print()
    print("============================================================")
    print("Benchmark summary")
    print("============================================================")
    print("Timing scope:         compute only")
    print("GPU timing method:    CUDA events")
    print("CPU timing method:    perf_counter")
    print("Transfers included:   no")
    print("JIT included:         no")
    print(f"Matrix shapes:        A={a_d.shape}, B={b_d.shape}, C={c_d.shape}")
    print(f"dtype:                {np.dtype(dtype)}")
    print(f"iterations:           {iters}")
    print(f"Useful scalar FLOPs:  {flops:.0f}")
    print(f"GPU elapsed mean:     {gpu_mean:.6e} s")
    print(f"GPU elapsed median:   {gpu_median:.6e} s")
    print(f"GPU elapsed minimum:  {gpu_min:.6e} s")
    print(f"GPU elapsed stddev:   {gpu_std:.6e} s")
    print(f"CPU elapsed mean:     {cpu_mean:.6e} s")
    print(f"CPU elapsed median:   {cpu_median:.6e} s")
    print(f"CPU elapsed minimum:  {cpu_min:.6e} s")
    print(f"CPU elapsed stddev:   {cpu_std:.6e} s")
    print(f"Compute-only speedup: {speedup:.2f}x")
    print(f"GPU GFLOP/s median:   {gpu_gflops_s:.4f}")
    print(f"CPU GFLOP/s median:   {cpu_gflops_s:.4f}")
    print(f"Check passed:         {check_passed}")
    return check_passed


def main():
    parser = argparse.ArgumentParser(
        description=("Numba CUDA shared-memory tiled matrix multiplication")
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
        default=10,
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

    if not cuda.is_available():
        raise RuntimeError("CUDA is not available")

    if args.dtype == "float32":
        dtype = np.float32
        kernel = matmul_shared_float32
    elif args.dtype == "float64":
        dtype = np.float64
        kernel = matmul_shared_float64
    else:
        raise TypeError("dtype must be either float32 or float64")

    n = args.n
    m = args.m
    l = args.l

    print(f"A shape: {n} x {l}")
    print(f"B shape: {l} x {m}")
    print(f"C shape: {n} x {m}")
    print(f"dtype:   {np.dtype(dtype)}")
    print(f"iters:   {args.iters}")

    rng = np.random.default_rng(args.seed)

    x_h = rng.standard_normal((n, l)).astype(dtype)
    y_h = rng.standard_normal((l, m)).astype(dtype)
    z_h = np.zeros((n, m), dtype=dtype)

    with nvtx_range("H2D memory copies", id_color=6):
        x_d = cuda.to_device(x_h)
        y_d = cuda.to_device(y_h)
        z_d = cuda.to_device(z_h)

    checks_passed = matmul(
        a_d=x_d,
        b_d=y_d,
        c_d=z_d,
        iters=args.iters,
        kernel=kernel,
        dtype=dtype,
    )
    return 0 if checks_passed else 1

if __name__ == "__main__":
    raise SystemExit(main())


