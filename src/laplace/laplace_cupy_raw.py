#!/usr/bin/env python3

import argparse
from contextlib import contextmanager
from time import perf_counter

import cupy as cp
import numpy as np

@contextmanager
def nvtx_range(name: str, id_color: int = -1):
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

# RawKernel assumes contiguous row-major arrays.
# Do not pass transposed or strided views unless the kernel explicitly handles strides.

jacobi_kernel = cp.RawKernel(r'''
extern "C" __global__
void jacobi_2d(
    const double* __restrict__ u,
    double* __restrict__ unew,
    const size_t nx,
    const size_t ny,
    const double dx2,
    const double dy2,
    const double inv_denom
)
{
    size_t j = blockDim.x * blockIdx.x + threadIdx.x + 1;
    size_t i = blockDim.y * blockIdx.y + threadIdx.y + 1;
    // Physical interior only.
    if (j >= nx - 1 || i >= ny - 1) {
        return;
    }
    const size_t idx = i * nx + j;
    const double left  = u[idx - 1];
    const double right = u[idx + 1];
    const double up    = u[idx - nx];
    const double down  = u[idx + nx];
    unew[idx] = ((left + right) * dy2 + (up + down) * dx2) * inv_denom;
}
''', 'jacobi_2d')


rel_update_norm_partial_kernel = cp.RawKernel(r'''
extern "C" __global__
void rel_update_norm_partial_2d(
    const double* __restrict__ u,
    const double* __restrict__ unew,
    double* __restrict__ partial_diff,
    double* __restrict__ partial_norm,
    const size_t nx,
    const size_t n
)
{
    extern __shared__ double shared[];
    double* sdiff = shared;
    double* snorm = shared + blockDim.x;
    const unsigned int tid = threadIdx.x;
    const size_t stride = (size_t)blockDim.x * (size_t)gridDim.x;
    const size_t ncols  = nx - 2;
    double diff_sum = 0.0;
    double norm_sum = 0.0;
    for (size_t k = (size_t)blockIdx.x * blockDim.x + tid; k < n; k += stride)
    {
        const size_t local_i = (k / ncols);
        const size_t local_j = (k - local_i * ncols);
        const size_t i = local_i + 1;
        const size_t j = local_j + 1;
        const size_t idx = i * nx + j;
        const double d = unew[idx] - u[idx];
        const double v = unew[idx];
        diff_sum += d * d;
        norm_sum += v * v;
    }
    sdiff[tid] = diff_sum;
    snorm[tid] = norm_sum;
    __syncthreads();
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            sdiff[tid] += sdiff[tid + s];
            snorm[tid] += snorm[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        partial_diff[blockIdx.x] = sdiff[0];
        partial_norm[blockIdx.x] = snorm[0];
    }
}
''', 'rel_update_norm_partial_2d')


class Laplace2DCuPyRawKernel:
    """
    Serial single-GPU CuPy Jacobi solver using a custom CUDA RawKernel.
    This is the RawKernel alternative to the vectorized Laplace2DCuPy class.
    Global layout:
        u[0, :]      bottom boundary
        u[-1, :]     top boundary
        u[:, 0]      left boundary
        u[:, -1]     right boundary
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
        dtype=cp.float64,
    ):
        Laplace2DCuPyRawKernel._validate_problem_parameters(nx, ny, lx, ly, bc_left, bc_right, bc_bottom, bc_top, dtype)
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
        self.dtype = dtype
        self.u = cp.zeros((self.ny, self.nx), dtype=self.dtype)
        self.unew = cp.zeros_like(self.u)
        self._apply_dirichlet_bc(self.u)
        self._apply_dirichlet_bc(self.unew)
        self.n_interior = (self.ny - 2) * (self.nx - 2)
        self.rel_update_norm_threads = 256
        if self.rel_update_norm_threads & (self.rel_update_norm_threads - 1) != 0:
            raise ValueError("rel_update_norm_threads must be a power of two")
        
        # Cap the number of partial sums. 
        # The grid-stride loop still processes every interior point.
        # The value 4096 is a heuristic.
        self.rel_update_norm_blocks = min(4096, (self.n_interior + self.rel_update_norm_threads - 1) // self.rel_update_norm_threads,)
        self.rel_update_norm_shared_bytes = (2 * self.rel_update_norm_threads * cp.dtype(cp.float64).itemsize)
        self.jacobi_threads = (32, 8)
        self.jacobi_blocks = (
            (self.nx - 2 + self.jacobi_threads[0] - 1) // self.jacobi_threads[0],
            (self.ny - 2 + self.jacobi_threads[1] - 1) // self.jacobi_threads[1],
        )
        self._validate_kernel_configuration()
        self.nx_kernel = np.uint64(self.nx)
        self.ny_kernel = np.uint64(self.ny)
        self.n_interior_kernel = np.uint64(self.n_interior)
        self.dx2_kernel = np.float64(self.dx2)
        self.dy2_kernel = np.float64(self.dy2)
        self.inv_denom_kernel = np.float64(self.inv_denom)
        self.partial_diff = cp.empty(self.rel_update_norm_blocks, dtype=cp.float64)
        self.partial_norm = cp.empty(self.rel_update_norm_blocks, dtype=cp.float64)


    @staticmethod
    def _validate_problem_parameters(nx, ny, lx, ly, bc_left, bc_right, bc_bottom, bc_top, dtype):
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
        if dtype != cp.float64:
            raise ValueError("This RawKernel implementation currently supports only cp.float64")

    @staticmethod
    def _validate_run_parameters(max_iters, tol, check_every, fixed_iters):
        integer_types = (int, np.integer)
        if not isinstance(max_iters, integer_types):
            raise TypeError("max_iters must be an integer")
        if not isinstance(check_every, integer_types):
            raise TypeError("check_every must be an integer")
        if (fixed_iters is not None and not isinstance(fixed_iters, integer_types)):
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

    def _validate_kernel_configuration(self):
        attributes = cp.cuda.Device().attributes
        jacobi_threads_per_block = (self.jacobi_threads[0] * self.jacobi_threads[1])
        if jacobi_threads_per_block > attributes["MaxThreadsPerBlock"]:
            raise ValueError(f"Jacobi block requires {jacobi_threads_per_block} threads. "
                             f"But the device supports only {attributes['MaxThreadsPerBlock']}")

        if self.jacobi_threads[0] > attributes["MaxBlockDimX"]:
            raise ValueError(f"Jacobi block X dimension {self.jacobi_threads[0]} exceeds MaxBlockDimX={attributes['MaxBlockDimX']}")

        if self.jacobi_threads[1] > attributes["MaxBlockDimY"]:
            raise ValueError(f"Jacobi block Y dimension {self.jacobi_threads[1]} exceeds MaxBlockDimY={attributes['MaxBlockDimY']}")

        if self.rel_update_norm_threads > attributes["MaxThreadsPerBlock"]:
            raise ValueError(f"Reduction block requires {self.rel_update_norm_threads} threads. "
                             f"But the device supports only {attributes['MaxThreadsPerBlock']}")
        
        if self.rel_update_norm_shared_bytes > attributes["MaxSharedMemoryPerBlock"]:
            raise ValueError(f"Reduction kernel requires {self.rel_update_norm_shared_bytes} bytes of shared memory per block. " 
                             f"But the device supports only {attributes['MaxSharedMemoryPerBlock']} bytes")

    # Boundary conditions
    def _apply_dirichlet_bc(self, arr):
        """
        Apply Dirichlet boundary conditions.
        Corners follow bottom/top assignment, matching the ordering used by
        the NumPy, vectorized CuPy, and distributed versions:
          1. left/right
          2. bottom/top
        """
        with nvtx_range("_apply_dirichlet_bc"):
            arr[ :,  0] = self.bc_left
            arr[ :, -1] = self.bc_right
            arr[ 0,  :] = self.bc_bottom
            arr[-1,  :] = self.bc_top


    def reset_fields(self):
        """
        Reset solution arrays to the initial condition.
        This is useful after warm-up so the timed run starts from the same state.
        """
        self.u.fill(0.0)
        self.unew.fill(0.0)
        self._apply_dirichlet_bc(self.u)
        self._apply_dirichlet_bc(self.unew)
        cp.cuda.get_current_stream().synchronize()


    def warmup(self):
        """
        Warm up RawKernel launch and reduction kernels.
        """
        with nvtx_range("warmup"):
            self.jacobi_step()
            _ = self.compute_relative_l2_update_norm()
            cp.cuda.get_current_stream().synchronize()
        self.reset_fields()

    # Compute kernel
    def jacobi_step(self):
        """
        One Jacobi update using the custom CUDA RawKernel.
        Updates only the true physical interior:
            unew[1:ny-1, 1:nx-1]
        Boundary values are initialized once in both arrays and are not overwritten
        because the kernel updates only interior points.
        """
        with nvtx_range("jacobi_step"):
            jacobi_kernel(self.jacobi_blocks, self.jacobi_threads,
                (self.u, self.unew, self.nx_kernel, self.ny_kernel, self.dx2_kernel, self.dy2_kernel, self.inv_denom_kernel,),
            )

    # Relative update norm
    def compute_relative_l2_update_norm(self):
        with nvtx_range("compute_relative_l2_update_norm"):
            rel_update_norm_partial_kernel((self.rel_update_norm_blocks,), (self.rel_update_norm_threads,),
                (self.u, self.unew, self.partial_diff, self.partial_norm, self.nx_kernel, self.n_interior_kernel),
                shared_mem=self.rel_update_norm_shared_bytes,
            )
            diff2 = cp.sum(self.partial_diff)
            norm2 = cp.sum(self.partial_norm)
            rel_update_norm_gpu = cp.sqrt(diff2) / cp.maximum(cp.sqrt(norm2), 1.0e-300)
            return float(rel_update_norm_gpu.get())

    # Solver loop
    def run(self, max_iters, tol, check_every, fixed_iters=None):
        rel_update_norm = float("inf")
        niters = 0
        with nvtx_range("run"):
            self._validate_run_parameters(max_iters, tol, check_every, fixed_iters)
            iteration_limit = fixed_iters if fixed_iters is not None else max_iters
            start = cp.cuda.Event()
            stop = cp.cuda.Event()
            cp.cuda.get_current_stream().synchronize()
            wall_t0 = perf_counter()
            start.record()
            for it in range(1, iteration_limit + 1):
                self.jacobi_step()
                do_check = (it == 1 or it % check_every == 0 or it == iteration_limit)
                converged = False
                if do_check:
                    rel_update_norm = self.compute_relative_l2_update_norm()
                    converged = (fixed_iters is None and rel_update_norm < tol)
                self.u, self.unew = self.unew, self.u
                niters = it
                if converged:
                    break
            stop.record()
            stop.synchronize()
            elapsed_cuda_event = cp.cuda.get_elapsed_time(start, stop) / 1000.0
            elapsed_wall = perf_counter() - wall_t0
        return niters, rel_update_norm, elapsed_cuda_event, elapsed_wall

    # Output
    def write_hdf5(self, filename, niters, rel_update_norm):
        """
        Write CuPy RawKernel solution to HDF5.
        The solution is copied explicitly from GPU to host before writing.
        """
        with nvtx_range("write_hdf5"):
            import h5py
            host_u = cp.asnumpy(self.u)
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
                dset.attrs["backend"] = "CuPy RawKernel"
                dset.attrs["niters"] = niters
                dset.attrs["rel_update_norm"] = float(rel_update_norm)
                dset.attrs["dtype"] = str(self.u.dtype)
                dset.attrs["decomposition"] = "serial/global array"
                dset.attrs["convergence_metric"] = ("relative L2 iterate-update norm over physical interior")
                dset.attrs["rel_update_norm_method"] = ("CUDA partial reduction + CuPy reduction")

# CLI
def parse_args():
    parser = argparse.ArgumentParser(
        description="Serial single-GPU CuPy RawKernel Laplace solver."
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
    ndev = cp.cuda.runtime.getDeviceCount()
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
    if args.device < 0:
        raise ValueError("device id must be non-negative")
    if args.device >= ndev:
        raise ValueError(f"device id {args.device} is out of range; visible devices: {ndev}")
    for name in ("bc_left", "bc_right", "bc_bottom", "bc_top"):
        value = getattr(args, name)
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")


def main():
    args = parse_args()
    validate_common_args(args)
    cp.cuda.Device(args.device).use()
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    raw_name = props["name"]
    name = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
    
    solver = Laplace2DCuPyRawKernel(
        nx=args.nx,
        ny=args.ny,
        lx=args.lx,
        ly=args.ly,
        bc_left=args.bc_left,
        bc_right=args.bc_right,
        bc_bottom=args.bc_bottom,
        bc_top=args.bc_top,
    )
    
    bytes_per_array = args.ny * args.nx * cp.dtype(cp.float64).itemsize
    partial_bytes = (2 * solver.rel_update_norm_blocks * np.dtype(np.float64).itemsize)
    persistent_field_bytes = 2 * bytes_per_array
    explicit_device_bytes = persistent_field_bytes + partial_bytes
    
    print("Serial CuPy RawKernel Laplace solver")
    print(f"  CUDA device        : {args.device}")
    print(f"  grid               : ny={args.ny}, nx={args.nx}")
    print(f"  physical size      : ly={args.ly}, lx={args.lx}")
    print(f"  check every        : {args.check_every}")
    print(f"  GPU device name    : {name}")
    print(f"  memory per field   : {bytes_per_array / 1024**2:.2f} MiB")
    print(f"  persistent fields  : {persistent_field_bytes / 1024**2:.2f} MiB")
    print(f"  reduction partials : {partial_bytes / 1024**2:.2f} MiB")
    print(f"  explicit total     : {explicit_device_bytes / 1024**2:.2f} MiB")
    print(f"  output             : {args.output}")
    if args.fixed_iters is not None:
        print(f"  mode               : fixed iteration")
        print(f"  fixed iterations   : {args.fixed_iters}")
    else:
        print(f"  mode               : tolerance")
        print(f"  tolerance          : {args.tol}")
        print(f"  max iterations     : {args.max_iters}")
    print("", flush=True)

    print("Warming up...", flush=True)
    solver.warmup()
    print("Warm-up complete.", flush=True)

    niters, rel_update_norm, elapsed_cuda_event, elapsed_wall = solver.run(
        max_iters=args.max_iters,
        tol=args.tol,
        check_every=args.check_every,
        fixed_iters=args.fixed_iters,
    )
    
    global_updates_per_iter = ((args.nx - 2) * (args.ny - 2))
    total_updates = global_updates_per_iter * niters
    gupdates_per_second_cuda_event = (total_updates / elapsed_cuda_event / 1.0e9 if elapsed_cuda_event > 0.0 else float("inf"))
    gupdates_per_second_wall       = (total_updates / elapsed_wall       / 1.0e9 if elapsed_wall       > 0.0 else float("inf"))
    converged = (args.fixed_iters is None and rel_update_norm < args.tol)
    if args.fixed_iters is None and not converged:
        print("Warning: maximum iteration count reached before the requested tolerance was satisfied.", flush=True,)
    print()
    print("============================================================")
    print("Solver summary")
    print("============================================================")
    print("Implementation                                         : CuPy RawKernel fused Jacobi kernel")
    print(f"CuPy RawKernel iterations                              : {niters}")
    if args.fixed_iters is None:
        print(f"CuPy RawKernel converged                               : {converged}")
    else:
        print("CuPy RawKernel convergence used for stopping           : no")
    print(f"CuPy RawKernel final relative iterate-update norm      : {rel_update_norm:.6e}")
    print(f"CuPy RawKernel CUDA-event solver interval [s]          : {elapsed_cuda_event:.6f}")
    print(f"CuPy RawKernel wall elapsed time [s]                   : {elapsed_wall:.6f}")
    print(f"CuPy RawKernel CUDA-event time per iter [ms]           : {1000.0 * elapsed_cuda_event / max(niters, 1):.6f}")
    print(f"CuPy RawKernel wall time per iter [ms]                 : {1000.0 * elapsed_wall / max(niters, 1):.6f}")
    print("Throughput metric: effective monitored solver")
    print(f"CuPy RawKernel CUDA-event throughput [GCell-updates/s] : {gupdates_per_second_cuda_event:.6f}")
    print(f"CuPy RawKernel wall throughput [GCell-updates/s]       : {gupdates_per_second_wall:.6f}")
    if args.output is not None:
        solver.write_hdf5(args.output, niters, rel_update_norm)
        print(f"CuPy RawKernel wrote solution                          : {args.output}")
    print("", flush=True)


if __name__ == "__main__":
    main()


