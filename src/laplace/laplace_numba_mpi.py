#!/usr/bin/env python3
import os
import argparse
import mpi4py
mpi4py.rc.thread_level = "funneled"
from mpi4py import MPI
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


def get_local_rank(mpi_comm):
    """
    Determine GPU-local rank.
    Priority:
      1. common launcher environment variables
      2. MPI shared-memory communicator rank
    """
    env_names = (
        "LOCAL_RANK",
        "SLURM_LOCALID",
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "MV2_COMM_WORLD_LOCAL_RANK",
    )
    for name in env_names:
        value = os.environ.get(name)
        if value is not None:
            return int(value)
    shared_comm = mpi_comm.Split_type(MPI.COMM_TYPE_SHARED)
    local_rank = shared_comm.Get_rank()
    shared_comm.Free()
    return local_rank


def ordered_print(mpi_comm, message):
    """
    Print one rank at a time, in MPI rank order.
    This is only for readable diagnostics, not numerical correctness.
    """
    rank = mpi_comm.Get_rank()
    size = mpi_comm.Get_size()
    for r in range(size):
        mpi_comm.Barrier()
        if rank == r:
            print(message, flush=True)
    mpi_comm.Barrier()


def decompose_1d(n_global, size, rank):
    """
    1D block decomposition with remainder.
    """
    counts = np.full(size, n_global // size, dtype=np.int64)
    counts[: n_global % size] += 1
    offsets = np.zeros(size + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)
    local_n = int(counts[rank])
    start = int(offsets[rank])
    end = int(offsets[rank + 1])
    return local_n, start, end, counts, offsets


@cuda.jit
def jacobi_kernel(u, unew, nx, i0, i1, dx2, dy2, inv_denom):
    """
    Numba CUDA Jacobi update.
    """
    j, local_i = cuda.grid(2)
    j += 1
    i = i0 + local_i
    if j >= nx - 1 or i >= i1:
        return
    left  = u[i, j - 1]
    right = u[i, j + 1]
    up    = u[i - 1, j]
    down  = u[i + 1, j]
    unew[i, j] = ((left + right) * dy2 + (up + down) * dx2) * inv_denom

@cuda.jit
def compute_diff_sq_kernel(u, unew, nx, i0, i1, out_diff, out_norm):
    """
    Compact 1D rel_update_norm map.
    Writes exactly the active local interior points:
        out_diff[k] = (unew[i, j] - u[i, j])**2
        out_norm[k] = unew[i, j]**2
    Only the first n entries are valid.
    """
    k = cuda.grid(1)
    ncols = nx - 2
    nrows = i1 - i0
    n = nrows * ncols
    if k < n:
        local_i = k // ncols
        local_j = k - local_i * ncols
        i = i0 + local_i
        j = 1 + local_j
        diff = unew[i, j] - u[i, j]
        val = unew[i, j]
        out_diff[k] = diff * diff
        out_norm[k] = val * val


class Laplace2DDistributedNumbaCUDAAwareMPI:
    """
    Distributed Jacobi solver for the 2D Laplace equation.
    Decomposition:
        1D row decomposition along global y direction.
    Local layout:
        d_u[0, :]                 upper halo
        d_u[1:local_ny+1, :]      owned physical rows
        d_u[local_ny+1, :]        lower halo
    Communication:
        CUDA-aware MPI Sendrecv for halos.
        CUDA-aware MPI Allreduce is performed on host scalars after
        Numba CUDA local reductions.
    Computation:
        Numba CUDA kernels.
    Relative update norm:
        Numba CUDA map kernel + cuda.reduce + MPI.Allreduce.
    """
    def __init__(
        self,
        nx,
        ny,
        lx,
        ly,
        mpi_comm,
        bc_left=0.0,
        bc_right=1.0,
        bc_bottom=0.0,
        bc_top=0.0,
    ):
        Laplace2DDistributedNumbaCUDAAwareMPI._validate_problem_parameters(nx, ny, lx, ly, bc_left, bc_right, bc_bottom, bc_top)
        self.nx = int(nx)
        self.ny = int(ny)
        self.lx = float(lx)
        self.ly = float(ly)
        self.mpi_comm = mpi_comm
        self.rank = mpi_comm.Get_rank()
        self.size = mpi_comm.Get_size()
        (
            self.local_ny,
            self.ystart,
            self.yend,
            self.counts,
            self.offsets,
        ) = decompose_1d(self.ny, self.size, self.rank)
        if self.local_ny <= 0:
            raise ValueError(
                "This example assumes each rank owns at least one row. "
                "Use number of ranks <= ny."
            )
        self.up = self.rank - 1 if self.rank > 0 else MPI.PROC_NULL
        self.down = self.rank + 1 if self.rank < self.size - 1 else MPI.PROC_NULL
        self.dx = self.lx / (self.nx - 1)
        self.dy = self.ly / (self.ny - 1)
        self.dx2 = self.dx * self.dx
        self.dy2 = self.dy * self.dy
        self.inv_denom = 1.0 / (2.0 * (self.dx2 + self.dy2))
        self.bc_left = float(bc_left)
        self.bc_right = float(bc_right)
        self.bc_bottom = float(bc_bottom)
        self.bc_top = float(bc_top)
        h_u = np.zeros((self.local_ny + 2, self.nx), dtype=np.float64)
        h_unew = np.zeros_like(h_u)
        self._apply_dirichlet_bc_host(h_u)
        self._apply_dirichlet_bc_host(h_unew)
        self.d_u = cuda.to_device(h_u)
        self.d_unew = cuda.to_device(h_unew)
        max_local_interior = self.local_ny * (self.nx - 2)
        self.d_diff_sq = cuda.device_array(max_local_interior, dtype=np.float64)
        self.d_norm_sq = cuda.device_array(max_local_interior, dtype=np.float64)
        self.h_partial = np.zeros(2, dtype=np.float64)
        self.h_global = np.zeros(2, dtype=np.float64)

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

    def _owned_update_row_bounds(self):
        """
        Return local row bounds for true update/rel_update_norm region.
        Excludes:
          - halo rows
          - global bottom boundary
          - global top boundary
        """
        first = 1
        last_exclusive = self.local_ny + 1
        if self.ystart == 0:
            first = 2
        if self.yend == self.ny:
            last_exclusive = self.local_ny
        return first, last_exclusive

    def _apply_dirichlet_bc_host(self, arr):
        """
        Apply physical Dirichlet boundary conditions to host array.
        Boundaries are initialized once in both arrays and are not overwritten
        because the CUDA kernel updates only interior points.
        """
        with nvtx_range("_apply_dirichlet_bc_host", id_color=1):
            arr[1 : self.local_ny + 1, 0] = self.bc_left
            arr[1 : self.local_ny + 1, self.nx - 1] = self.bc_right
            if self.rank == 0:
                arr[1, :] = self.bc_bottom
            if self.rank == self.size - 1:
                arr[self.local_ny, :] = self.bc_top

    def exchange_halos_mpi(self):
        """
        Exchange halo rows using CUDA-aware MPI Sendrecv.
        Same convention as the CuPy MPI version:
          - tag 100: upward send, rank r sends row 1 to rank r-1
          - tag 200: downward send, rank r sends row local_ny to rank r+1
        """
        with nvtx_range("exchange_halos_mpi", id_color=2):
            # Required for portable CUDA-aware MPI behavior:
            # make sure previous GPU writes to d_u are visible to MPI.
            cuda.default_stream().synchronize()
            # Exchange with upper neighbor:
            # rank r sends owned first row to r-1
            # rank r receives upper halo from r-1
            self.mpi_comm.Sendrecv(
                sendbuf=self.d_u[1, :],
                dest=self.up,
                sendtag=100,
                recvbuf=self.d_u[0, :],
                source=self.up,
                recvtag=200,
            )
            # Exchange with lower neighbor:
            # rank r sends owned last row to r+1
            # rank r receives lower halo from r+1
            self.mpi_comm.Sendrecv(
                sendbuf=self.d_u[self.local_ny, :],
                dest=self.down,
                sendtag=200,
                recvbuf=self.d_u[self.local_ny + 1, :],
                source=self.down,
                recvtag=100,
            )

    def jacobi_step(self):
        """
        One Numba CUDA Jacobi update.
        """
        with nvtx_range("jacobi_step", id_color=3):
            i0, i1 = self._owned_update_row_bounds()
            if i0 >= i1:
                return
            threads = (TPB_X, TPB_Y)
            blocks = ((self.nx - 2 + threads[0] - 1) // threads[0], (i1 - i0 + threads[1] - 1) // threads[1],)
            jacobi_kernel[blocks, threads](
                self.d_u, self.d_unew, self.nx, i0, i1, self.dx2, self.dy2, self.inv_denom,
            )

    def compute_relative_l2_update_norm(self):
        """
        Compute global relative L2 update rel_update_norm:
            ||u_new - u||_2 / ||u_new||_2
        over true global interior points only.
        """
        rel_update_norm = 0.0
        with nvtx_range("jacobi_l2", id_color=4):
            i0, i1 = self._owned_update_row_bounds()
            local_diff2 = 0.0
            local_norm2 = 0.0
            if i0 < i1:
                n = (i1 - i0) * (self.nx - 2)
                blocks = (n + REDUCE_TPB - 1) // REDUCE_TPB
                compute_diff_sq_kernel[blocks, REDUCE_TPB](
                    self.d_u,
                    self.d_unew,
                    self.nx,
                    i0,
                    i1,
                    self.d_diff_sq,
                    self.d_norm_sq,
                )
                # Only first n entries are valid.
                local_diff2 = sum_reduce(self.d_diff_sq, size=n)
                local_norm2 = sum_reduce(self.d_norm_sq, size=n)
            self.h_partial[0] = float(local_diff2)
            self.h_partial[1] = float(local_norm2)
            self.mpi_comm.Allreduce(self.h_partial, self.h_global, op=MPI.SUM)
            diff2 = self.h_global[0]
            norm2 = self.h_global[1]
            rel_update_norm = np.sqrt(diff2) / max(np.sqrt(norm2), 1.0e-300)
        return float(rel_update_norm)



    def reset_fields(self):
        h_u = np.zeros((self.local_ny + 2, self.nx), dtype=np.float64)
        h_unew = np.zeros_like(h_u)
        self._apply_dirichlet_bc_host(h_u)
        self._apply_dirichlet_bc_host(h_unew)
        self.d_u.copy_to_device(h_u)
        self.d_unew.copy_to_device(h_unew)
        cuda.synchronize()
        self.mpi_comm.Barrier()


    def warmup(self):
        """
        Warm up CUDA kernels, reductions, and MPI halo exchange.
        Resets fields afterwards.
        """
        with nvtx_range("warmup", id_color=5):
            self.mpi_comm.Barrier()
            self.exchange_halos_mpi()
            self.jacobi_step()
            _ = self.compute_relative_l2_update_norm()
            cuda.synchronize()
            self.mpi_comm.Barrier()
        self.reset_fields()


    def run(self, max_iters, tol, check_every, fixed_iters=None):
        """
        Run Jacobi iteration.
        """
        rel_update_norm = np.inf
        niters = 0
        elapsed = 0.0 
        with nvtx_range("jacobi_run", id_color=6):
            Laplace2DDistributedNumbaCUDAAwareMPI._validate_run_parameters(max_iters, tol, check_every, fixed_iters)
            nstop = fixed_iters if fixed_iters is not None else max_iters
            self.mpi_comm.Barrier()
            start = cuda.event()
            stop = cuda.event()
            cuda.synchronize()
            start.record()
            for it in range(1, nstop + 1):
                self.exchange_halos_mpi()
                self.jacobi_step()
                do_check = (it == 1 or it % check_every == 0 or it == nstop)
                if do_check:
                    rel_update_norm = self.compute_relative_l2_update_norm()
                    if fixed_iters is None and rel_update_norm < tol:
                        # Keep the converged field in d_u for output.
                        self.d_u, self.d_unew = self.d_unew, self.d_u
                        niters = it
                        break
                self.d_u, self.d_unew = self.d_unew, self.d_u
                niters = it
            stop.record()
            stop.synchronize()
            self.mpi_comm.Barrier()
            elapsed_local = cuda.event_elapsed_time(start, stop) / 1000.0
            elapsed = self.mpi_comm.allreduce(elapsed_local, op=MPI.MAX)
        return niters, rel_update_norm, elapsed

    def write_parallel_hdf5(self, filename, niters=None, rel_update_norm=None):
        """
        Write global solution using parallel HDF5.
        Note:
            HDF5 write uses host memory. We explicitly copy local GPU slab
            to NumPy before entering HDF5.
        """
        with nvtx_range("jacobi_write", id_color=7):
            import h5py 
            if not h5py.get_config().mpi:
                raise RuntimeError(
                    "This h5py build does not have MPI support. "
                    "Rebuild h5py against parallel HDF5 with HDF5_MPI=ON."
                )
            self.mpi_comm.Barrier()
            cuda.synchronize()
            local_host = self.d_u.copy_to_host()[1 : self.local_ny + 1, :]
            with h5py.File(filename, "w", driver="mpio", comm=self.mpi_comm) as f:
                dset = f.create_dataset(
                    "u",
                    shape=(self.ny, self.nx),
                    dtype=local_host.dtype,
                )
                dset[self.ystart : self.yend, :] = local_host
                dset.attrs["nx"] = self.nx
                dset.attrs["ny"] = self.ny
                dset.attrs["lx"] = self.lx
                dset.attrs["ly"] = self.ly
                dset.attrs["dx"] = self.dx
                dset.attrs["dy"] = self.dy
                dset.attrs["dtype"] = str(local_host.dtype)
                dset.attrs["mpi_size"] = self.size
                dset.attrs["rank_decomposition"] = "rows/y"
                dset.attrs["bc_left"] = self.bc_left
                dset.attrs["bc_right"] = self.bc_right
                dset.attrs["bc_bottom"] = self.bc_bottom
                dset.attrs["bc_top"] = self.bc_top
                dset.attrs["solver"] = "Jacobi"
                dset.attrs["backend"] = "Numba CUDA + CUDA-aware MPI + mpi4py"
                dset.attrs["decomposition"] = "1D row decomposition"
                dset.attrs["communication"] = "CUDA-aware MPI Sendrecv halos"
                dset.attrs["rel_update_norm_method"] = ("compact 1D map kernel + cuda.reduce + MPI.Allreduce")
                if niters is not None:
                    dset.attrs["niters"] = niters
                if rel_update_norm is not None:
                    dset.attrs["rel_update_norm"] = rel_update_norm
            self.mpi_comm.Barrier()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Distributed 2D Laplace solver using Numba CUDA, "
            "CUDA-aware MPI halo exchange, mpi4py, and parallel HDF5."
        )
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
    parser.add_argument("--output", type=str, default="laplace_numba_mpi.h5")
    parser.add_argument("--fixed-iters", type=int, default=None,
        help="Run exactly this many iterations, ignoring tolerance.",
    )
    return parser.parse_args()


def validate_common_args(args, mpi_comm):
    rank = mpi_comm.Get_rank()
    size = mpi_comm.Get_size()
    if args.nx < 3 or args.ny < 3:
        if rank == 0:
            print("Error: nx and ny must both be at least 3", flush=True)
        mpi_comm.Abort(1)
    if not np.isfinite(args.lx) or not np.isfinite(args.ly):
        if rank == 0:
            print("Error: lx and ly must be finite", flush=True)
        mpi_comm.Abort(1)
    if args.lx <= 0.0 or args.ly <= 0.0:
        if rank == 0:
            print("Error: lx and ly must be positive", flush=True)
        mpi_comm.Abort(1)
    if size > args.ny - 2:
        if rank == 0:
            print("Error: number of MPI ranks must be <= ny - 2, but boundary-only ranks are possible with full-row decomposition", flush=True)
        mpi_comm.Abort(1)
    if args.max_iters < 1:
        if rank == 0:
            print("Error: max_iters must be at least 1", flush=True)
        mpi_comm.Abort(1)
    if args.check_every < 1:
        if rank == 0:
            print("Error: check_every must be at least 1", flush=True)
        mpi_comm.Abort(1)
    if args.fixed_iters is not None and args.fixed_iters < 1:
        if rank == 0:
            print("Error: fixed_iters must be at least 1", flush=True)
        mpi_comm.Abort(1)
    if args.fixed_iters is None:
        if not np.isfinite(args.tol) or args.tol <= 0.0:
            if rank == 0:
                print("Error: tol must be finite and positive unless fixed_iters is used", flush=True)
            mpi_comm.Abort(1)
    for name in ("bc_left", "bc_right", "bc_bottom", "bc_top"):
        value = getattr(args, name)
        if not np.isfinite(value):
            if rank == 0:
                print(f"Error: {name} must be finite", flush=True)
            mpi_comm.Abort(1)

def main():
    args = parse_args()
    mpi_comm = MPI.COMM_WORLD
    validate_common_args(args, mpi_comm)
    rank = mpi_comm.Get_rank()
    size = mpi_comm.Get_size()
    local_rank = get_local_rank(mpi_comm)
    try:
        ndev = len(cuda.gpus)
    except Exception as exc:
        raise RuntimeError("Could not query CUDA devices") from exc
    if ndev <= 0:
        raise RuntimeError("No CUDA devices found")
    if local_rank >= ndev:
        print(
            f"Error on rank {rank}: local_rank={local_rank} "
            f"but only {ndev} CUDA devices are visible",
            flush=True,
        )
        mpi_comm.Abort(1)
    cuda.select_device(local_rank)
    solver = Laplace2DDistributedNumbaCUDAAwareMPI(
        nx=args.nx,
        ny=args.ny,
        lx=args.lx,
        ly=args.ly,
        mpi_comm=mpi_comm,
        bc_left=args.bc_left,
        bc_right=args.bc_right,
        bc_bottom=args.bc_bottom,
        bc_top=args.bc_top,
    )
    if rank == 0:
        print("Distributed Numba CUDA / CUDA-aware MPI Laplace solver")
        print(f"  global grid       : ny={args.ny}, nx={args.nx}")
        print(f"  physical size     : ly={args.ly}, lx={args.lx}")
        print(f"  MPI ranks         : {size}")
        print(f"  halo mode         : CUDA-aware MPI")
        print(f"  check every       : {args.check_every}")
        print(f"  output            : {args.output}")
        if args.fixed_iters is not None:
            print(f"  mode              : fixed iteration")
            print(f"  fixed iterations  : {args.fixed_iters}")
        else:
            print(f"  mode              : tolerance")
            print(f"  tolerance         : {args.tol}")
            print(f"  max iterations    : {args.max_iters}")
        print("", flush=True)
    device = cuda.get_current_device()
    device_id = local_rank
    gpu_name = (
        device.name.decode()
        if isinstance(device.name, bytes)
        else str(device.name)
    )
    local_bytes_per_array  = ((solver.local_ny + 2) * solver.nx * np.dtype(np.float64).itemsize)
    global_bytes_per_array = (args.ny * args.nx * np.dtype(np.float64).itemsize)
    ordered_print(
        mpi_comm,
        (
            f"rank {rank:4d}/{size} | "
            f"local_rank={local_rank:2d} | "
            f"cuda_device={device_id:2d} | "
            f"gpu={gpu_name} | "
            f"global rows [{solver.ystart}:{solver.yend}) | "
            f"local_ny={solver.local_ny} | "
            f"local array={solver.d_u.shape} | "
            f"memory/field={local_bytes_per_array / 1024**2:.2f} MiB"
        ),
    )
    if rank == 0:
        print("", flush=True)
        print(f"  global field size : {global_bytes_per_array / 1024**2:.2f} MiB")
        print("", flush=True)

    if rank == 0:
        print("Warming up Numba CUDA + MPI...", flush=True)
    solver.warmup()
    if rank == 0:
        print("Warm-up complete.", flush=True)

    niters, rel_update_norm, elapsed = solver.run(
        max_iters=args.max_iters,
        tol=args.tol,
        check_every=args.check_every,
        fixed_iters=args.fixed_iters,
    )
    
    if args.output is not None:
        solver.write_parallel_hdf5(args.output, niters, rel_update_norm)
    
    global_updates_per_iter = (args.nx - 2) * (args.ny - 2)
    gupdates_per_second = (global_updates_per_iter * niters / elapsed / 1.0e9 )

    if rank == 0:
        print("", flush=True)
        print(f"Numba+MPI iterations                     : {niters}")
        print(f"Numba+MPI final relative update norm     : {rel_update_norm:.6e}")
        print(f"Numba+MPI elapsed time [s]               : {elapsed:.6f}")
        print(f"Numba+MPI time per iter [ms]             : {1000.0 * elapsed / max(niters, 1):.6f}")
        print(f"Numba+MPI Jacobi throughput [GUpdates/s] : {gupdates_per_second:.6f}")
        print(f"Numba+MPI wrote solution                 : {args.output}")
        print("", flush=True)


if __name__ == "__main__":
    main()


