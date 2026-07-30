import argparse
import math
from time import perf_counter
import numpy as np
from numba import cuda, float32, float64
from contextlib import contextmanager
import cupy as cp

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



# Controls threads per block and shared memory usage.
# The computation will be done on blocks of TPBxTPB elements.
# TPB should not be larger than 32 in this example
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

    threads = tpb_x * tpb_y

    if threads > 1024:
        raise ValueError(
            f"CUDA block has {threads} threads, but the maximum is 1024"
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
def fast_matmul_float64(A, B, C):
    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float64)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float64)
    b_c = cuda.blockIdx.x
    b_r = cuda.blockIdx.y
    t_c = cuda.threadIdx.x
    t_r = cuda.threadIdx.y
    col = b_c * TPB + t_c
    row = b_r * TPB + t_r
    
    # The dot product is chunked into dot products of TPB-long vectors.
    Cvalue = float64(0.0)
    num_tiles = (A.shape[1] + TPB - 1) // TPB
    
    for tile in range(num_tiles):
        col_A = tile * TPB + t_c
        row_B = tile * TPB + t_r
        
        # Preload data into shared memory
        if row < A.shape[0] and col_A < A.shape[1]:
            sA[t_r, t_c]= A[row, col_A]  # copy A to shared mem
        else:
            sA[t_r, t_c]= float64(0.)
            
        if row_B < B.shape[0] and col < B.shape[1]:
            sB[t_r, t_c]= B[row_B, col]  # copy B to shared mem
        else:
            sB[t_r, t_c]= float64(0.)
            
        # Wait until all threads finish preloading
        cuda.syncthreads()
        
        # Computes partial product on the shared memory
        for k in range(TPB):
            Cvalue += sA[t_r, k] * sB[k, t_c]
            
        # Wait until all threads finish computing
        cuda.syncthreads()
        
    if row < A.shape[0] and col < B.shape[1]:
        C[row, col] = Cvalue

       
@cuda.jit
def fast_matmul_float32(A, B, C):
    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    b_c = cuda.blockIdx.x
    b_r = cuda.blockIdx.y
    t_c = cuda.threadIdx.x
    t_r = cuda.threadIdx.y
    col = b_c * TPB + t_c
    row = b_r * TPB + t_r
    
    # The dot product is chunked into dot products of TPB-long vectors.
    Cvalue = float32(0.0)
    num_tiles = (A.shape[1] + TPB - 1) // TPB
    
    for tile in range(num_tiles):
        col_A = tile * TPB + t_c
        row_B = tile * TPB + t_r
        
        # Preload data into shared memory
        if row < A.shape[0] and col_A < A.shape[1]:
            sA[t_r, t_c]= A[row, col_A]  # copy A to shared mem
        else:
            sA[t_r, t_c]= float32(0.)
            
        if row_B < B.shape[0] and col < B.shape[1]:
            sB[t_r, t_c]= B[row_B, col]  # copy B to shared mem
        else:
            sB[t_r, t_c]= float32(0.)
            
        # Wait until all threads finish preloading
        cuda.syncthreads()
        
        # Computes partial product on the shared memory
        for k in range(TPB):
            Cvalue += sA[t_r, k] * sB[k, t_c]
            
        # Wait until all threads finish computing
        cuda.syncthreads()
        
    if row < A.shape[0] and col < B.shape[1]:
        C[row, col] = Cvalue



def compute_launch_geometry(a, b):
    threadsperblock = (TPB, TPB)

    rows_c = a.shape[0]
    cols_c = b.shape[1]

    blockspergrid_x = math.ceil(cols_c / TPB)
    blockspergrid_y = math.ceil(rows_c / TPB)

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



def gpu_kernel_time_seconds(kernel, blockspergrid, threadsperblock, a, b, c):
    start_event = cuda.event()
    stop_event = cuda.event()
    cuda.synchronize()
    start_event.record()
    
    with nvtx_range("GPU tiled shared-memory matmul", id_color=1):
        kernel[blockspergrid, threadsperblock](a, b, c)

    stop_event.record()
    stop_event.synchronize()
    return cuda.event_elapsed_time(start_event, stop_event) * 1.0e-3


def matmul(
    a_d,
    b_d,
    c_d,
    iters: int,
    kernel,
    dtype: np.dtype,
) -> None:
    validate_positive_int("iters", iters)
    validate_matrix_shapes(a_d, b_d, c_d)
    
    # FIX: Use the global TPB instead of undefined tpb_x and tpb_y
    validate_block_size(TPB, TPB)

    blockspergrid, threadsperblock = compute_launch_geometry(
        a_d,
        b_d,
    )

    print(f"blockspergrid   = {blockspergrid}")
    print(f"threadsperblock = {threadsperblock}")

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

    print("Warmup complete. Timings below exclude JIT compilation.")

    timings_gpu = np.empty(iters, dtype=np.float64)
    timings_cpu = np.empty(iters, dtype=np.float64)

    rtol, atol = dtype_tolerances(dtype)

    for i in range(iters):
        # ----------------------------------------------------
        # GPU timing using CUDA events
        # ----------------------------------------------------
        timings_gpu[i] = gpu_kernel_time_seconds(
            kernel=kernel,
            blockspergrid=blockspergrid,
            threadsperblock=threadsperblock,
            a=a_d,
            b=b_d,
            c=c_d,
        )


        # ----------------------------------------------------
        # CPU timing with preallocated output
        # ----------------------------------------------------
        cpu_t0 = perf_counter()

        with nvtx_range("CPU NumPy matmul", id_color=4):
            np.matmul(a_h, b_h, out=check_h)

        cpu_t1 = perf_counter()
        timings_cpu[i] = cpu_t1 - cpu_t0

        # ----------------------------------------------------
        # Correctness check
        # ----------------------------------------------------
        with nvtx_range("D2H result copy", id_color=5):
            c_h = c_d.copy_to_host()

        ok = np.allclose(c_h, check_h, rtol=rtol, atol=atol)
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
    cpu_mean = float(timings_cpu.mean())

    n = a_d.shape[0]
    l = a_d.shape[1]
    m = b_d.shape[1]

    flops = flop_count_matmul(n, l, m)
    gflops = flops * 1.0e-9
    gpu_gflops_s = gflops / gpu_mean
    cpu_gflops_s = gflops / cpu_mean

    print()
    print("============================================================")
    print("Benchmark summary")
    print("============================================================")
    print(f"Matrix shapes:       A={a_d.shape}, B={b_d.shape}, C={c_d.shape}")
    print(f"dtype:               {dtype}")
    print(f"iterations:          {iters}")
    print(f"FLOPs per matmul:    {flops:.0f}")
    print(f"GPU elapsed mean:    {gpu_mean:.6e} s")
    print(f"CPU elapsed mean:    {cpu_mean:.6e} s")
    print(f"Speedup:             {cpu_mean / gpu_mean:.2f}x")
    print(f"GPU GFLOP/s:         {gpu_gflops_s:.4f}")
    print(f"CPU GFLOP/s:         {cpu_gflops_s:.4f}")


def main():
    parser = argparse.ArgumentParser()
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

    if not cuda.is_available():
        raise RuntimeError("CUDA is not available")

    if args.dtype == "float32":
        dtype = np.float32
        kernel = fast_matmul_float32
    elif args.dtype == "float64":
        dtype = np.float64
        kernel = fast_matmul_float64
    else:
        raise TypeError("dtype must be either float32 or float64")

    n = args.n
    m = args.m
    l = args.l

    print(f"A shape: {n} x {l}")
    print(f"B shape: {l} x {m}")
    print(f"C shape: {n} x {m}")
    print(f"dtype:   {dtype}")
    print(f"iters:   {args.iters}")

    rng = np.random.default_rng(args.seed)

    x_h = rng.standard_normal((n, l)).astype(dtype)
    y_h = rng.standard_normal((l, m)).astype(dtype)
    z_h = np.zeros((n, m), dtype=dtype)

    with nvtx_range("H2D memory copies", id_color=6):
        x_d = cuda.to_device(x_h)
        y_d = cuda.to_device(y_h)
        z_d = cuda.to_device(z_h)

    matmul(
        a_d=x_d,
        b_d=y_d,
        c_d=z_d,
        iters=args.iters,
        kernel=kernel,
        dtype=dtype,
    )


if __name__ == "__main__":
    main()
