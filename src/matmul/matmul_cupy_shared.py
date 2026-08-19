#!/usr/bin/env python3

import argparse
from contextlib import contextmanager
from time import perf_counter

import cupy as cp
import numpy as np


# ============================================================
# DEFAULTS
# ============================================================

DEFAULT_TPB = 16


# ============================================================
# CUDA C++ SOURCE: SHARED-MEMORY TILED MATMUL
# ============================================================

CUDA_SOURCE = r'''
template <typename T, int TILE>
__global__ void mm_kernel_tiled(
    const size_t A_num_rows,
    const size_t B_num_cols,
    const size_t A_num_cols,
    const T* __restrict__ A,
    const T* __restrict__ B,
    T* __restrict__ C
) {
    __shared__ T Atile[TILE][TILE];
    __shared__ T Btile[TILE][TILE];

    const size_t tx = static_cast<size_t>(threadIdx.x);
    const size_t ty = static_cast<size_t>(threadIdx.y);

    const size_t col = static_cast<size_t>(blockIdx.x) * TILE + tx;
    const size_t row = static_cast<size_t>(blockIdx.y) * TILE + ty;

    T acc = static_cast<T>(0);

    const size_t num_tiles = (A_num_cols + TILE - 1) / TILE;

    for (size_t tile = 0; tile < num_tiles; ++tile) {
        const size_t a_col = tile * TILE + tx;
        const size_t b_row = tile * TILE + ty;

        if (row < A_num_rows && a_col < A_num_cols) {
            Atile[ty][tx] = A[row * A_num_cols + a_col];
        } else {
            Atile[ty][tx] = static_cast<T>(0);
        }

        if (b_row < A_num_cols && col < B_num_cols) {
            Btile[ty][tx] = B[b_row * B_num_cols + col];
        } else {
            Btile[ty][tx] = static_cast<T>(0);
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += Atile[ty][k] * Btile[k][tx];
        }

        __syncthreads();
    }

    if (row < A_num_rows && col < B_num_cols) {
        C[row * B_num_cols + col] = acc;
    }
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


#def validate_tpb(tpb: int) -> None:
#    validate_positive_int("tpb", tpb)
#
#    threads_per_block = tpb * tpb
#
#    if threads_per_block > 1024:
#        raise ValueError(
#            f"tpb={tpb} gives {threads_per_block} threads/block, "
#            "but CUDA allows at most 1024"
#        )


def validate_tpb(tpb: int) -> None:
    validate_positive_int("tpb", tpb)

    device = cp.cuda.Device()
    attributes = device.attributes

    threads_per_block = tpb * tpb
    max_threads_per_block = attributes["MaxThreadsPerBlock"]
    max_block_dim_x = attributes["MaxBlockDimX"]
    max_block_dim_y = attributes["MaxBlockDimY"]

    if threads_per_block > max_threads_per_block:
        raise ValueError(
            f"tpb={tpb} gives {threads_per_block} threads/block, "
            f"but the device supports at most "
            f"{max_threads_per_block}"
        )

    if tpb > max_block_dim_x:
        raise ValueError(
            f"tpb={tpb} exceeds MaxBlockDimX={max_block_dim_x}"
        )

    if tpb > max_block_dim_y:
        raise ValueError(
            f"tpb={tpb} exceeds MaxBlockDimY={max_block_dim_y}"
        )


def shared_memory_bytes(tpb: int, dtype: np.dtype) -> int:
    return 2 * tpb * tpb * np.dtype(dtype).itemsize


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

    if not a.flags.c_contiguous:
        raise ValueError("A must be C-contiguous")
    if not b.flags.c_contiguous:
        raise ValueError("B must be C-contiguous")
    if not c.flags.c_contiguous:
        raise ValueError("C must be C-contiguous")



def compute_launch_geometry(a: cp.ndarray, b: cp.ndarray, tpb: int):
    rows_c = a.shape[0]
    cols_c = b.shape[1]
    
    blocks_x = (cols_c + tpb - 1) // tpb
    blocks_y = (rows_c + tpb - 1) // tpb

    grid = (blocks_x, blocks_y)
    block = (tpb, tpb)

    return grid, block



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
        return 1.0e-10, 1.0e-10

    raise TypeError("dtype must be either float32 or float64")


def compile_kernels(tpb: int):
    """
    Compile templated CUDA kernel specializations for the requested tile size.
    """
    name_float = f"mm_kernel_tiled<float, {tpb}>"
    name_double = f"mm_kernel_tiled<double, {tpb}>"

    module = cp.RawModule(
        code=CUDA_SOURCE,
        options=("-std=c++17", "--generate-line-info"),
        name_expressions=(name_float, name_double),
    )

    ker_float = module.get_function(name_float)
    ker_double = module.get_function(name_double)

    return ker_float, ker_double


def launch_kernel(
    kernel,
    grid,
    block,
    a_d: cp.ndarray,
    b_d: cp.ndarray,
    c_d: cp.ndarray,
) -> None:
    kernel(
        grid,
        block,
        (
            np.uint64(a_d.shape[0]),
            np.uint64(b_d.shape[1]),
            np.uint64(a_d.shape[1]),
            a_d,
            b_d,
            c_d,
        ),
    )


def gpu_kernel_time_seconds(
    kernel,
    grid,
    block,
    a_d: cp.ndarray,
    b_d: cp.ndarray,
    c_d: cp.ndarray,
    start_event,
    stop_event,
) -> float:
    """
    Time one CUDA kernel launch using CUDA events.

    CuPy reports event elapsed time in milliseconds.
    """

    start_event.record()

    with nvtx_range("Launch CuPy tiled RawKernel matmul"):
        launch_kernel(kernel, grid, block, a_d, b_d, c_d)

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

# ============================================================
# BENCHMARK DRIVER
# ============================================================

def matmul(
    a_d: cp.ndarray,
    b_d: cp.ndarray,
    c_d: cp.ndarray,
    iters: int,
    kernel,
    tpb: int,
    dtype: np.dtype,
) -> bool:
    validate_positive_int("iters", iters)
    validate_tpb(tpb)
    validate_matrix_shapes(a_d, b_d, c_d)
    validate_selected_dtype(a_d, b_d, c_d, dtype)
    grid, block = compute_launch_geometry(a_d, b_d, tpb)
    
    shared_bytes = shared_memory_bytes(tpb, dtype)
    device = cp.cuda.Device()
    max_shared_bytes = device.attributes["MaxSharedMemoryPerBlock"]
    if shared_bytes > max_shared_bytes:
        raise ValueError(
            f"Tile size {tpb}x{tpb} with dtype={np.dtype(dtype)} "
            f"requires {shared_bytes} bytes of shared memory per block, "
            f"but the device supports only {max_shared_bytes} bytes"
        )

    print(f"blockspergrid   = {grid}")
    print(f"threadsperblock = {block}")
    print(f"tile size       = {tpb} x {tpb}")
    print(f"shared memory   = {shared_bytes} bytes/block")
    
    # Copy inputs once for CPU reference.
    with nvtx_range("D2H input copies"):
        a_h = cp.asnumpy(a_d)
        b_h = cp.asnumpy(b_d)

    # Preallocate CPU reference output.
    check_h = np.empty((a_h.shape[0], b_h.shape[1]), dtype=dtype)

    # --------------------------------------------------------
    # Warm-up launch.
    #
    # The warm-up excludes compilation, module loading, and
    # first-launch setup from the measured iterations. 
    # --------------------------------------------------------
    print("Warming up CUDA tiled kernel...")

    with nvtx_range("Warmup tiled shared-memory matmul"):
        launch_kernel(kernel, grid, block, a_d, b_d, c_d)

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

    for i in range(iters):
        # ----------------------------------------------------
        # GPU timing with CUDA events
        # ----------------------------------------------------
        timings_gpu[i] = gpu_kernel_time_seconds(
            kernel=kernel,
            grid=grid,
            block=block,
            a_d=a_d,
            b_d=b_d,
            c_d=c_d,
            start_event=start_event,
            stop_event=stop_event,
        )

    # ----------------------------------------------------
    # CPU timing with preallocated output
    # ----------------------------------------------------
    for i in range(iters):
        with nvtx_range("CPU NumPy matmul"):
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
    print("Compilation included: no")
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

# ============================================================
# MAIN
# ============================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "CuPy RawModule shared-memory tiled matrix multiplication benchmark"
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
        "--tpb",
        type=int,
        default=DEFAULT_TPB,
        help=(
            "Tile size and CUDA block dimension. "
            "Block shape is tpb x tpb."
        ),
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
    validate_positive_int("tpb", args.tpb)
    if cp.cuda.runtime.getDeviceCount() <= 0:
        raise RuntimeError("No CUDA device found")
    validate_tpb(args.tpb)

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
    tpb = args.tpb

    print(f"A shape: {n} x {l}")
    print(f"B shape: {l} x {m}")
    print(f"C shape: {n} x {m}")
    print(f"dtype:   {dtype_np}")
    print(f"iters:   {args.iters}")
    print(f"tpb:     {tpb}")

    cp.random.seed(args.seed)

    with nvtx_range("GPU allocation and initialization"):
        x_d = cp.random.standard_normal((n, l), dtype=dtype_cp)
        y_d = cp.random.standard_normal((l, m), dtype=dtype_cp)
        z_d = cp.zeros((n, m), dtype=dtype_cp)

    print("Compiling RawModule tiled kernels...")
    ker_float, ker_double = compile_kernels(tpb)
    print("Compilation complete.")

    if dtype_np == np.dtype(np.float32):
        kernel = ker_float
    elif dtype_np == np.dtype(np.float64):
        kernel = ker_double
    else:
        raise TypeError("dtype must be either float32 or float64")

    check_passed = matmul(
        a_d=x_d,
        b_d=y_d,
        c_d=z_d,
        iters=args.iters,
        kernel=kernel,
        tpb=tpb,
        dtype=dtype_np,
    )

    return 0 if check_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

