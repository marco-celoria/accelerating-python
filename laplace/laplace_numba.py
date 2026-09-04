#!/usr/bin/env python3
import argparse
from contextlib import contextmanager
from time import perf_counter

import numpy as np
from numba import cuda, float64

try:
    import cupy as cp
except ImportError:
    cp = None


@contextmanager
def nvtx_range(name: str, id_color: int = -1):
    if cp is None:
        yield
        return

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


# A full warp spans x, the contiguous row-major dimension.
# The 32 x 8 block contains 256 threads.
TPB_X = 32
TPB_Y = 8
REDUCE_TPB = 256
REDUCE_BLOCKS_PER_SM = 4

@cuda.reduce
def sum_reduce(a, b):
    return a + b


@cuda.jit
def jacobi_kernel(u, unew, nx, ny, dx2, dy2, inv_denom):
    """
    Perform one Jacobi update over the physical interior.

    The kernel assumes contiguous row-major arrays and updates:
        unew[1:ny-1, 1:nx-1]
    """
    j, i = cuda.grid(2)
    j += 1
    i += 1
    if j >= nx - 1 or i >= ny - 1:
        return
    left  = u[i, j - 1]
    right = u[i, j + 1]
    down  = u[i - 1, j]
    up    = u[i + 1, j]
    unew[i, j] = ((left + right) * dy2 + (up + down) * dx2) * inv_denom


@cuda.jit
def rel_update_norm_partial_2d_kernel(u, unew, nx, n, partial_diff, partial_norm):
    """
    Partial reduction kernel for the relative L2 update norm over the physical interior.
    Accumulates local sums in shared memory per thread block and writes partial block sums.
    """
    sdiff = cuda.shared.array(shape=REDUCE_TPB, dtype=float64)
    snorm = cuda.shared.array(shape=REDUCE_TPB, dtype=float64)

    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    bdim = cuda.blockDim.x
    gdim = cuda.gridDim.x

    stride = bdim * gdim
    ncols = nx - 2

    diff_sum = 0.0
    norm_sum = 0.0

    # Grid-stride loop over physical interior points
    for k in range(bid * bdim + tid, n, stride):
        local_i = k // ncols
        local_j = k - local_i * ncols
        i = 1 + local_i
        j = 1 + local_j

        diff = unew[i, j] - u[i, j]
        val  = unew[i, j]

        diff_sum += diff * diff
        norm_sum += val * val

    sdiff[tid] = diff_sum
    snorm[tid] = norm_sum
    cuda.syncthreads()

    # In-block tree reduction
    s = bdim // 2
    while s > 0:
        if tid < s:
            sdiff[tid] += sdiff[tid + s]
            snorm[tid] += snorm[tid + s]
        cuda.syncthreads()
        s //= 2

    # Write block results to global partial arrays
    if tid == 0:
        partial_diff[bid] = sdiff[0]
        partial_norm[bid] = snorm[0]


class Laplace2DNumbaCUDA:
    """
    Serial single-GPU Numba CUDA Jacobi solver using an in-block shared memory
    partial reduction for the relative L2 update norm.
    """
    def __init__(
        self,
        nx,
        ny,
        lx,
        ly,
        bc_left=0.0,
        bc_right=1.0,
        bc_bottom=0.0,
        bc_top=0.0,
    ):
        Laplace2DNumbaCUDA._validate_problem_parameters(nx, ny, lx, ly, bc_left, bc_right, bc_bottom, bc_top)
        self.nx = int(nx)
        self.ny = int(ny)
        self.lx = float(lx)
        self.ly = float(ly)
        self.dx = self.lx / (self.nx - 1)
        self.dy = self.ly / (self.ny - 1)
        self.dx2 = self.dx * self.dx
        self.dy2 = self.dy * self.dy
        self.inv_denom = 1.0 / (2.0 * (self.dx2 + self.dy2))
        self.bc_left = float(bc_left)
        self.bc_right = float(bc_right)
        self.bc_bottom = float(bc_bottom)
        self.bc_top = float(bc_top)
        self.h_initial = np.zeros((self.ny, self.nx), dtype=np.float64)
        self._apply_dirichlet_bc_host(self.h_initial)
        self.d_u = cuda.to_device(self.h_initial)
        self.d_unew = cuda.to_device(self.h_initial)
        self.jacobi_threads = (TPB_X, TPB_Y)
        self.jacobi_blocks = (
            (self.nx - 2 + self.jacobi_threads[0] - 1) // self.jacobi_threads[0],
            (self.ny - 2 + self.jacobi_threads[1] - 1) // self.jacobi_threads[1],
        )
        self.n_interior = (self.ny - 2) * (self.nx - 2)
        device = cuda.get_current_device()
        self.num_sms = device.MULTIPROCESSOR_COUNT
        self.norm_blocks = max(1, min((self.n_interior + REDUCE_TPB - 1) // REDUCE_TPB, REDUCE_BLOCKS_PER_SM * self.num_sms,),) 
        self._validate_kernel_configuration()

        # Compact partial workspace allocated only per thread-block
        self.d_partial_diff = cuda.device_array(self.norm_blocks, dtype=np.float64)
        self.d_partial_norm = cuda.device_array(self.norm_blocks, dtype=np.float64)

    def _validate_kernel_configuration(self):
        device = cuda.get_current_device()
        jacobi_threads_per_block = TPB_X * TPB_Y
        if REDUCE_TPB < 1 or (REDUCE_TPB & (REDUCE_TPB - 1)) != 0:
            raise ValueError("REDUCE_TPB must be a positive power of two")
        if REDUCE_BLOCKS_PER_SM < 1:
            raise ValueError("REDUCE_BLOCKS_PER_SM must be positive")
        if jacobi_threads_per_block > device.MAX_THREADS_PER_BLOCK:
            raise ValueError(f"Jacobi block requires {jacobi_threads_per_block} threads. "
                             f"But the device supports only {device.MAX_THREADS_PER_BLOCK}")
        if TPB_X > device.MAX_BLOCK_DIM_X:
            raise ValueError(f"TPB_X={TPB_X} exceeds MAX_BLOCK_DIM_X={device.MAX_BLOCK_DIM_X}")
        if TPB_Y > device.MAX_BLOCK_DIM_Y:
            raise ValueError(f"TPB_Y={TPB_Y} exceeds MAX_BLOCK_DIM_Y={device.MAX_BLOCK_DIM_Y}")
        if REDUCE_TPB > device.MAX_THREADS_PER_BLOCK:
            raise ValueError(f"REDUCE_TPB={REDUCE_TPB} exceeds MAX_THREADS_PER_BLOCK={device.MAX_THREADS_PER_BLOCK}")

    @staticmethod
    def _validate_problem_parameters(nx, ny, lx, ly, bc_left, bc_right, bc_bottom, bc_top):
        if not isinstance(nx, (int, np.integer)) or not isinstance(ny, (int, np.integer)):
            raise TypeError("nx and ny must be integers")
        if int(nx) < 3 or int(ny) < 3:
            raise ValueError("nx and ny must both be at least 3")
        values = {
            "lx": lx,
            "ly": ly,
            "bc_left": bc_left,
            "bc_right": bc_right,
            "bc_bottom": bc_bottom,
            "bc_top": bc_top,
        }
        for name, value in values.items():
            if not np.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if float(lx) <= 0.0 or float(ly) <= 0.0:
            raise ValueError("lx and ly must be positive")

    @staticmethod
    def _validate_run_parameters(
        max_iters,
        tol,
        check_every,
        fixed_iters,
    ):
        integer_types = (int, np.integer)
        if not isinstance(max_iters, integer_types):
            raise TypeError("max_iters must be an integer")
        if not isinstance(check_every, integer_types):
            raise TypeError("check_every must be an integer")
        if fixed_iters is not None and not isinstance(fixed_iters, integer_types):
            raise TypeError("fixed_iters must be an integer")
        if max_iters < 1:
            raise ValueError("max_iters must be at least 1")
        if check_every < 1:
            raise ValueError("check_every must be at least 1")
        if fixed_iters is not None and fixed_iters < 1:
            raise ValueError("fixed_iters must be at least 1")
        if fixed_iters is None:
            if not np.isfinite(float(tol)) or float(tol) <= 0.0:
                raise ValueError("tol must be finite and positive unless fixed_iters is used")

    def _apply_dirichlet_bc_host(self, arr):
        with nvtx_range("_apply_dirichlet_bc_host", id_color=1):
            arr[:,  0] = self.bc_left
            arr[:, -1] = self.bc_right
            arr[ 0, :] = self.bc_bottom
            arr[-1, :] = self.bc_top

    def reset_fields(self):
        self.d_u.copy_to_device(self.h_initial)
        self.d_unew.copy_to_device(self.h_initial)
        cuda.synchronize()

    def jacobi_step(self):
        with nvtx_range("jacobi_step", id_color=2):
            jacobi_kernel[self.jacobi_blocks, self.jacobi_threads](
                self.d_u, self.d_unew, self.nx, self.ny, self.dx2, self.dy2, self.inv_denom,
            )

    def compute_relative_l2_update_norm(self):
        with nvtx_range("jacobi_l2", id_color=3):
            rel_update_norm_partial_2d_kernel[self.norm_blocks, REDUCE_TPB](
                self.d_u,
                self.d_unew,
                self.nx,
                self.n_interior,
                self.d_partial_diff,
                self.d_partial_norm,
            )
            diff2 = sum_reduce(self.d_partial_diff, size=self.norm_blocks)
            norm2 = sum_reduce(self.d_partial_norm, size=self.norm_blocks)
            rel_update_norm = np.sqrt(diff2) / max(np.sqrt(norm2), 1.0e-300)
            return float(rel_update_norm)

    def warmup(self):
        with nvtx_range("warmup", id_color=4):
            self.jacobi_step()
            _ = self.compute_relative_l2_update_norm()
            cuda.synchronize()
        self.reset_fields()

    def run(self, max_iters, tol, check_every, fixed_iters=None):
        rel_update_norm = float("inf")
        niters = 0
        with nvtx_range("jacobi_run", id_color=5):
            self._validate_run_parameters(max_iters, tol, check_every, fixed_iters)
            iteration_limit = fixed_iters if fixed_iters is not None else max_iters
            start = cuda.event()
            stop = cuda.event()
            cuda.synchronize()
            wall_t0 = perf_counter()
            start.record()
            for it in range(1, iteration_limit + 1):
                self.jacobi_step()
                do_check = (it == 1 or it % check_every == 0 or it == iteration_limit)
                converged = False
                if do_check:
                    rel_update_norm = self.compute_relative_l2_update_norm()
                    converged = (fixed_iters is None and rel_update_norm < tol)
                self.d_u, self.d_unew = self.d_unew, self.d_u
                niters = it
                if converged:
                    break
            stop.record()
            stop.synchronize()
            elapsed_cuda_event = cuda.event_elapsed_time(start, stop) / 1000.0
            elapsed_wall = perf_counter() - wall_t0
        return niters, rel_update_norm, elapsed_cuda_event, elapsed_wall

    def write_hdf5(self, filename, niters, rel_update_norm):
        with nvtx_range("jacobi_write", id_color=6):
            import h5py
            import os
            out_dir = os.path.dirname(filename)
            if out_dir:  
                os.makedirs(out_dir, exist_ok=True)
            host_u = self.d_u.copy_to_host()
            with h5py.File(filename, "w") as f:
                dset = f.create_dataset("u", data=host_u)
                dset.attrs["nx"] = self.nx
                dset.attrs["ny"] = self.ny
                dset.attrs["lx"] = self.lx
                dset.attrs["ly"] = self.ly
                dset.attrs["dx"] = self.dx
                dset.attrs["dy"] = self.dy
                dset.attrs["bc_left"] = self.bc_left
                dset.attrs["bc_right"] = self.bc_right
                dset.attrs["bc_bottom"] = self.bc_bottom
                dset.attrs["bc_top"] = self.bc_top
                dset.attrs["solver"] = "Jacobi"
                dset.attrs["backend"] = "Numba CUDA"
                dset.attrs["niters"] = niters
                dset.attrs["rel_update_norm"] = float(rel_update_norm)
                dset.attrs["dtype"] = str(host_u.dtype)
                dset.attrs["rel_update_norm_method"] = ("Numba CUDA block-shared partial reduction kernel + cuda.reduce")
                dset.attrs["decomposition"] = "serial/global array"
                dset.attrs["convergence_metric"] = ("relative L2 iterate-update norm over physical interior")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Serial single-GPU Numba CUDA Laplace solver."
    )
    parser.add_argument("--nx", type=int, default=1024)
    parser.add_argument("--ny", type=int, default=1024)
    parser.add_argument("--lx", type=float, default=1.0)
    parser.add_argument("--ly", type=float, default=1.0)
    parser.add_argument("--max-iters", type=int, default=20000)
    parser.add_argument("--tol", type=float, default=1.0e-8)
    parser.add_argument("--check-every", type=int, default=100)
    parser.add_argument("--bc-left", type=float, default=0.0)
    parser.add_argument("--bc-right", type=float, default=1.0)
    parser.add_argument("--bc-bottom", type=float, default=0.0)
    parser.add_argument("--bc-top", type=float, default=0.0)
    parser.add_argument("--fixed-iters", type=int, default=None,
        help=("Run exactly this many iterations without tolerance-based termination. "
              "The update norm is still evaluated according to --check-every."),
    )
    parser.add_argument("--device", type=int, default=0,
        help="CUDA device id.",
    )
    parser.add_argument("--output", type=str, default=None,
        help="Optional HDF5 filename.",
    )
    return parser.parse_args()


def validate_common_args(args):
    if not cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if args.nx < 3 or args.ny < 3:
        raise ValueError("nx and ny must both be at least 3")
    if not np.isfinite(args.lx) or not np.isfinite(args.ly):
        raise ValueError("lx and ly must be finite")
    if args.lx <= 0.0 or args.ly <= 0.0:
        raise ValueError("lx and ly must be positive")
    if args.max_iters < 1:
        raise ValueError("max_iters must be at least 1")
    if args.check_every < 1:
        raise ValueError("check_every must be at least 1")
    if args.fixed_iters is not None and args.fixed_iters < 1:
        raise ValueError("fixed_iters must be at least 1")
    if args.fixed_iters is None:
        if not np.isfinite(args.tol) or args.tol <= 0.0:
            raise ValueError("tol must be finite and positive unless fixed_iters is used")
    for name in ("bc_left", "bc_right", "bc_bottom", "bc_top"):
        value = getattr(args, name)
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if args.device < 0:
        raise ValueError("device id must be non-negative")
    if args.device >= len(cuda.gpus):
        raise ValueError(f"device id {args.device} is out of range")


def main():
    args = parse_args()
    validate_common_args(args)
    cuda.select_device(args.device)
    solver = Laplace2DNumbaCUDA(
        nx=args.nx,
        ny=args.ny,
        lx=args.lx,
        ly=args.ly,
        bc_left=args.bc_left,
        bc_right=args.bc_right,
        bc_bottom=args.bc_bottom,
        bc_top=args.bc_top,
    )
    
    device = cuda.get_current_device()
    gpu_name = (device.name.decode() if isinstance(device.name, bytes) else str(device.name))
    field_bytes = args.ny * args.nx * np.dtype(np.float64).itemsize
    norm_workspace_bytes = (2 * solver.norm_blocks * np.dtype(np.float64).itemsize)
    total_device_bytes = 2 * field_bytes + norm_workspace_bytes
    print("Serial Numba CUDA Laplace solver")
    print(f"  GPU device id         : {args.device}")
    print(f"  grid                  : ny={args.ny}, nx={args.nx}")
    print(f"  physical size         : ly={args.ly}, lx={args.lx}")
    print(f"  check every           : {args.check_every}")
    print(f"  GPU device name       : {gpu_name}")
    print(f"  GPU SM count          : {solver.num_sms}")
    print(f"  reduction threads     : {REDUCE_TPB}")
    print(f"  reduction blocks      : {solver.norm_blocks}")
    print(f"  reduction blocks/SM   : {REDUCE_BLOCKS_PER_SM}")
    print(f"  memory per field      : {field_bytes / 1024**2:.2f} MiB")
    print(f"  norm workspace        : {norm_workspace_bytes / 1024:.2f} KiB")
    print(f"  explicit device total : {total_device_bytes / 1024**2:.2f} MiB")
    print(f"  output                : {args.output}")
    if args.fixed_iters is not None:
        print(f"  mode                  : fixed iteration")
        print(f"  fixed iterations      : {args.fixed_iters}")
    else:
        print(f"  mode                  : tolerance")
        print(f"  tolerance             : {args.tol}")
        print(f"  max iterations        : {args.max_iters}")
    print("", flush=True)

    print("Warming up Numba CUDA kernels...", flush=True)
    solver.warmup()
    print("Warm-up complete.", flush=True)

    niters, rel_update_norm, elapsed_cuda_event, elapsed_wall = solver.run(
        max_iters=args.max_iters,
        tol=args.tol,
        check_every=args.check_every,
        fixed_iters=args.fixed_iters,
    )

    converged = (args.fixed_iters is None and rel_update_norm < args.tol)
    if args.fixed_iters is None and not converged:
        print("Warning: maximum iteration count reached before the requested tolerance was satisfied.", flush=True)
    global_updates_per_iter = ((args.nx - 2) * (args.ny - 2))
    total_updates = global_updates_per_iter * niters
    gupdates_per_second_cuda_event = (total_updates / elapsed_cuda_event / 1.0e9 if elapsed_cuda_event > 0.0 else float("inf"))
    gupdates_per_second_wall       = (total_updates / elapsed_wall       / 1.0e9 if elapsed_wall       > 0.0 else float("inf"))
    print()
    print("============================================================")
    print("Solver summary")
    print("============================================================")
    print("Implementation                                : Numba CUDA fused Jacobi kernel") 
    print(f"Numba CUDA iterations                          : {niters}")
    if args.fixed_iters is None:
        print(f"Numba CUDA converged                           : {converged}")
    else:
        print("Numba CUDA convergence used for stopping      : no")
    print(f"Numba CUDA final relative iterate-update norm : {rel_update_norm:.6e}")
    print(f"Numba CUDA-event solver interval [s]          : {elapsed_cuda_event:.6f}")
    print(f"Numba CUDA wall elapsed time [s]              : {elapsed_wall:.6f}")
    print(f"Numba CUDA-event time per iter [ms]            : {1000.0 * elapsed_cuda_event / max(niters, 1):.6f}")
    print(f"Numba CUDA wall time per iter [ms]             : {1000.0 * elapsed_wall / max(niters, 1):.6f}")
    print("Throughput metric: effective monitored solver")
    print(f"Numba CUDA-event throughput [GCell-updates/s] : {gupdates_per_second_cuda_event:.6f}")
    print(f"Numba wall throughput [GCell-updates/s]       : {gupdates_per_second_wall:.6f}")
    if args.output is not None:
        solver.write_hdf5(args.output, niters, rel_update_norm)
        print(f"Numba CUDA wrote solution                     : {args.output}")
    print("", flush=True)


if __name__ == "__main__":
    main()
