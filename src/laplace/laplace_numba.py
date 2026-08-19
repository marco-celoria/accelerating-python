#!/usr/bin/env python3
import argparse
import numpy as np
from numba import cuda
from contextlib import contextmanager
import cupy as cp


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



TPB_X = 32
TPB_Y = 8
REDUCE_TPB = 256

@cuda.reduce
def sum_reduce(a, b):
    return a + b


@cuda.jit
def jacobi_kernel(u, unew, nx, ny, dx2, dy2, inv_denom):
    """
    Numba CUDA Jacobi update.
    The kernel assumes contiguous row-major arrays.
    Updates only:
        unew[i0:i1, 1:nx-1]
    where i0 and i1 are local row bounds.
    """
    j, i = cuda.grid(2)
    j += 1
    i += 1
    if j >= nx - 1 or i >= ny - 1:
        return
    left  = u[i, j - 1]
    right = u[i, j + 1]
    up    = u[i - 1, j]
    down  = u[i + 1, j]
    unew[i, j] = (((left + right) * dy2 + (up + down) * dx2) * inv_denom)


@cuda.jit
def rel_update_norm_map_kernel(u, unew, nx, ny, out_diff, out_norm):
    """
    Compact rel_update_norm map over the global physical interior.
    It writes:
        out_diff[k] = (unew[i, j] - u[i, j])**2
        out_norm[k] = unew[i, j]**2
    for all interior points:
        i = 1..ny-2
        j = 1..nx-2
    """
    k = cuda.grid(1)
    ncols = nx - 2
    nrows = ny - 2
    n = nrows * ncols
    if k < n:
        local_i = k // ncols
        local_j = k - local_i * ncols
        i = 1 + local_i
        j = 1 + local_j
        diff = unew[i, j] - u[i, j]
        val  = unew[i, j]
        out_diff[k] = diff * diff
        out_norm[k] = val * val


class Laplace2DNumbaCUDA:
    """
    Serial single-GPU Numba CUDA Jacobi solver.
    This is the Numba CUDA equivalent of the CuPy RawKernel version.
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
        h_u = np.zeros((self.ny, self.nx), dtype=np.float64)
        h_unew = np.zeros_like(h_u)
        self._apply_dirichlet_bc_host(h_u)
        self._apply_dirichlet_bc_host(h_unew)
        self.d_u = cuda.to_device(h_u)
        self.d_unew = cuda.to_device(h_unew)
        n_interior = (self.ny - 2) * (self.nx - 2)
        self.d_diff_sq = cuda.device_array(n_interior, dtype=np.float64)
        self.d_norm_sq = cuda.device_array(n_interior, dtype=np.float64)

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
    def _validate_run_parameters(max_iters, tol, check_every, fixed_iters):
        if not isinstance(max_iters, int):
            raise TypeError("max_iters must be an integer")
        if not isinstance(check_every, int):
            raise TypeError("check_every must be an integer")
        if fixed_iters is not None and not isinstance(fixed_iters, int):
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

    # Boundary conditions
    def _apply_dirichlet_bc_host(self, arr):
        """
        Apply Dirichlet boundary conditions on a host array.
        Corners follow bottom/top assignment, matching the ordering used by
        the NumPy, CuPy, RawKernel, and distributed versions:
          1. left/right
          2. bottom/top
        """
        with nvtx_range("_apply_dirichlet_bc_host", id_color=1):
            arr[:,  0] = self.bc_left
            arr[:, -1] = self.bc_right
            arr[ 0, :] = self.bc_bottom
            arr[-1, :] = self.bc_top


    def reset_fields(self):
        """
        Reset device solution arrays to the initial condition.
        """
        h_u = np.zeros((self.ny, self.nx), dtype=np.float64)
        h_unew = np.zeros_like(h_u)
        self._apply_dirichlet_bc_host(h_u)
        self._apply_dirichlet_bc_host(h_unew)
        self.d_u.copy_to_device(h_u)
        self.d_unew.copy_to_device(h_unew)
        cuda.synchronize()


    # Compute kernel
    def jacobi_step(self):
        """
        One Jacobi update using a Numba CUDA kernel.
        Updates only the true physical interior:
            d_unew[1:ny-1, 1:nx-1]
        Boundary values are initialized once in both arrays and are not
        overwritten because the kernel updates only interior points.
        """
        with nvtx_range("jacobi_step", id_color=2):
            threads = (TPB_X, TPB_Y)
            blocks = ((self.nx - 2 + threads[0] - 1) // threads[0], (self.ny - 2 + threads[1] - 1) // threads[1],)
            jacobi_kernel[blocks, threads](
                self.d_u, self.d_unew, self.nx, self.ny, self.dx2, self.dy2, self.inv_denom,
            )

    # Relative update norm
    def compute_relative_l2_update_norm(self):
        """
        Compute relative L2 update norm:
            ||u_new - u||_2 / ||u_new||_2
        over true global interior points only.
        """
        rel_update_norm = 0.0
        with nvtx_range("jacobi_l2", id_color=3):
            n = (self.ny - 2) * (self.nx - 2)
            blocks = (n + REDUCE_TPB - 1) // REDUCE_TPB
            rel_update_norm_map_kernel[blocks, REDUCE_TPB](
                self.d_u,
                self.d_unew,
                self.nx,
                self.ny,
                self.d_diff_sq,
                self.d_norm_sq,
            )
            diff2 = sum_reduce(self.d_diff_sq, size=n)
            norm2 = sum_reduce(self.d_norm_sq, size=n)
            rel_update_norm = np.sqrt(diff2) / max(np.sqrt(norm2), 1.0e-300)
        return float(rel_update_norm) 


    def warmup(self):
        """
        Warm up Numba CUDA kernels and reductions.
        This forces JIT compilation before timing.
        """
        with nvtx_range("warmup", id_color=4):
            self.jacobi_step()
            _ = self.compute_relative_l2_update_norm()
            cuda.synchronize()
        self.reset_fields()

    # Solver loop
    def run(self, max_iters, tol, check_every, fixed_iters=None):
        rel_update_norm = float("inf")
        niters = 0
        elapsed = 0.0
        with nvtx_range("jacobi_run", id_color=5):
            Laplace2DNumbaCUDA._validate_run_parameters(max_iters, tol, check_every, fixed_iters)
            nstop = fixed_iters if fixed_iters is not None else max_iters
            start = cuda.event()
            stop = cuda.event()
            cuda.synchronize()
            start.record()
            for it in range(1, nstop + 1):
                self.jacobi_step()
                do_check = (it == 1 or it % check_every == 0 or it == nstop)
                if do_check:
                    rel_update_norm = self.compute_relative_l2_update_norm()
                    if fixed_iters is None and rel_update_norm < tol:
                        self.d_u, self.d_unew = self.d_unew, self.d_u
                        niters = it
                        break
                self.d_u, self.d_unew = self.d_unew, self.d_u
                niters = it
            stop.record()
            stop.synchronize()
            elapsed = cuda.event_elapsed_time(start, stop) / 1000.0
        return niters, rel_update_norm, elapsed

    # Output
    def write_hdf5(self, filename, niters, rel_update_norm):
        """
        Write Numba CUDA solution to HDF5.
        The solution is copied explicitly from GPU to host before writing.
        """
        with nvtx_range("jacobi_write", id_color=6):
            import h5py
            cuda.synchronize()
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

# CLI
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
        help="Run exactly this many iterations, ignoring tolerance.",
    )
    parser.add_argument("--device", type=int, default=0,
        help="CUDA device id.",
    )
    parser.add_argument("--output", type=str, default=None,
        help="Optional HDF5 filename.",
    )
    return parser.parse_args()


def validate_common_args(args):
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
    device = cuda.get_current_device()
    
    gpu_name = (
        device.name.decode()
        if isinstance(device.name, bytes)
        else str(device.name)
    )
    bytes_per_array = args.ny * args.nx * np.dtype(np.float64).itemsize
    print("Serial Numba CUDA Laplace solver")
    print(f"  CUDA device       : {args.device}")
    print(f"  grid              : ny={args.ny}, nx={args.nx}")
    print(f"  physical size     : ly={args.ly}, lx={args.lx}")
    print(f"  check every       : {args.check_every}")
    print(f"  GPU device name   : {gpu_name}")
    print(f"  memory per field  : {bytes_per_array / 1024**2:.2f} MiB")
    print(f"  memory total      : {2 * bytes_per_array / 1024**2:.2f} MiB")
    print(f"  output            : {args.output}")
    if args.fixed_iters is not None:
        print(f"  mode              : fixed iteration")
        print(f"  fixed iterations  : {args.fixed_iters}")
    else:
        print(f"  mode              : tolerance")
        print(f"  tolerance         : {args.tol}")
        print(f"  max iterations    : {args.max_iters}")
    print("", flush=True)
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

    print("Warming up Numba CUDA kernels...", flush=True)
    solver.warmup()
    print("Warm-up complete.", flush=True)

    niters, rel_update_norm, elapsed = solver.run(
        max_iters=args.max_iters,
        tol=args.tol,
        check_every=args.check_every,
        fixed_iters=args.fixed_iters,
    )

    global_updates_per_iter = (args.nx - 2) * (args.ny - 2)
    gupdates_per_second = (global_updates_per_iter * niters / elapsed / 1.0e9 )

    print(f"Numba iterations                     : {niters}")
    print(f"Numba final relative update norm     : {rel_update_norm:.6e}")
    print(f"Numba elapsed time [s]               : {elapsed:.6f}")
    print(f"Numba time per iter [ms]             : {1000.0 * elapsed / max(niters, 1):.6f}")
    print(f"Numba Jacobi throughput [GUpdates/s] : {gupdates_per_second:.6f}")
    if args.output is not None:
        solver.write_hdf5(args.output, niters, rel_update_norm)
        print(f"Numba wrote solution                 : {args.output}")
    print("", flush=True)


if __name__ == "__main__":
    main()


