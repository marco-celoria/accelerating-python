#!/usr/bin/env python3
import os
import argparse
import mpi4py
mpi4py.rc.thread_level = "funneled"
from mpi4py import MPI
import numpy as np
from numba import cuda, float64
from contextlib import contextmanager

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


TPB_X = 32
TPB_Y = 8
REDUCE_TPB = 256
REDUCE_BLOCKS_PER_SM = 4

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
    down  = u[i - 1, j]
    up    = u[i + 1, j]
    unew[i, j] = ((left + right) * dy2 + (up + down) * dx2) * inv_denom


@cuda.jit
def shared_reduce_norm_kernel(u, unew, nx, i0, i1, out_partial_diff, out_partial_norm):
    """
    Shared-memory block reduction using a grid-stride loop.
    Each block produces exactly one partial sum written to global memory.
    """
    # Allocate shared memory for the block (must be a compile-time constant size)
    s_diff = cuda.shared.array(shape=REDUCE_TPB, dtype=float64)
    s_norm = cuda.shared.array(shape=REDUCE_TPB, dtype=float64)

    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    bdim = cuda.blockDim.x

    ncols = nx - 2
    nrows = i1 - i0
    n = nrows * ncols

    # Grid-stride loop indexing
    stride = bdim * cuda.gridDim.x
    k = bid * bdim + tid

    diff_sum = 0.0
    norm_sum = 0.0

    # Each thread accumulates local partial sums spanning across the grid
    while k < n:
        local_i = k // ncols
        local_j = k - local_i * ncols
        i = i0 + local_i
        j = 1 + local_j

        diff = unew[i, j] - u[i, j]
        val = unew[i, j]

        diff_sum += diff * diff
        norm_sum += val * val

        k += stride

    # Write thread-local accumulations into shared memory
    s_diff[tid] = diff_sum
    s_norm[tid] = norm_sum
    cuda.syncthreads()

    # Block-wide tree reduction in shared memory
    s = bdim // 2
    while s > 0:
        if tid < s:
            s_diff[tid] += s_diff[tid + s]
            s_norm[tid] += s_norm[tid + s]
        cuda.syncthreads()
        s >>= 1

    # Thread 0 writes the block's reduced sum to global memory
    if tid == 0:
        out_partial_diff[bid] = s_diff[0]
        out_partial_norm[bid] = s_norm[0]


class Laplace2DDistributedNumbaCUDAAwareMPI:
    """
    Distributed Jacobi solver for the 2D Laplace equation.
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
            
        self.down = self.rank - 1 if self.rank > 0 else MPI.PROC_NULL
        self.up   = self.rank + 1 if self.rank < self.size - 1 else MPI.PROC_NULL
        self.dx   = self.lx / (self.nx - 1)
        self.dy   = self.ly / (self.ny - 1)
        self.dx2  = self.dx * self.dx
        self.dy2  = self.dy * self.dy
        self.inv_denom = 1.0 / (2.0 * (self.dx2 + self.dy2))
        self.bc_left   = float(bc_left)
        self.bc_right  = float(bc_right)
        self.bc_bottom = float(bc_bottom)
        self.bc_top    = float(bc_top)
        
        self.update_i0, self.update_i1 = (self._owned_update_row_bounds())
        self.local_update_count = (max(0, self.update_i1 - self.update_i0) * (self.nx - 2))

        self.jacobi_threads = (TPB_X, TPB_Y)
        if self.update_i0 < self.update_i1:
            self.jacobi_blocks = (
                (self.nx - 2 + self.jacobi_threads[0] - 1) // self.jacobi_threads[0],
                (self.update_i1 - self.update_i0 + self.jacobi_threads[1] - 1) // self.jacobi_threads[1],)
        else:
            self.jacobi_blocks = None
        
        # Select a rank-local reduction grid.
        # The grid-stride loop covers all local update cells while the
        # SM-based cap limits partial sums and second-stage reduction work.
        device = cuda.current_context().device
        self.num_sms = device.MULTIPROCESSOR_COUNT
        if self.local_update_count > 0:
            required_blocks = (self.local_update_count + REDUCE_TPB - 1) // REDUCE_TPB
            self.norm_blocks = min(required_blocks, REDUCE_BLOCKS_PER_SM * self.num_sms)
        else:
            self.norm_blocks = 0
        
        self._validate_kernel_configuration()
        
        # We only need `norm_blocks` elements allocated instead of `local_update_count` elements
        self.norm_workspace_size = max(1, self.norm_blocks)
        self.d_partial_diff = cuda.device_array(self.norm_workspace_size, dtype=np.float64,)
        self.d_partial_norm = cuda.device_array(self.norm_workspace_size, dtype=np.float64,)
        
        self.h_initial = np.zeros((self.local_ny + 2, self.nx), dtype=np.float64,)
        self._apply_dirichlet_bc_host(self.h_initial)
        self.d_u = cuda.to_device(self.h_initial)
        self.d_unew = cuda.to_device(self.h_initial)
        
        self.h_partial = np.zeros(2, dtype=np.float64)
        self.h_global = np.zeros(2, dtype=np.float64)


    def _validate_kernel_configuration(self):
        device = cuda.current_context().device
        jacobi_threads_per_block = (TPB_X * TPB_Y)
        if jacobi_threads_per_block > device.MAX_THREADS_PER_BLOCK:
            raise ValueError(f"Jacobi block requires {jacobi_threads_per_block} threads. "
                             f"But the device supports only {device.MAX_THREADS_PER_BLOCK}")
        if TPB_X > device.MAX_BLOCK_DIM_X:
            raise ValueError(f"TPB_X={TPB_X} exceeds MAX_BLOCK_DIM_X={device.MAX_BLOCK_DIM_X}")
        if TPB_Y > device.MAX_BLOCK_DIM_Y:
            raise ValueError(f"TPB_Y={TPB_Y} exceeds MAX_BLOCK_DIM_Y={device.MAX_BLOCK_DIM_Y}")
        if REDUCE_TPB > device.MAX_THREADS_PER_BLOCK:
            raise ValueError(f"REDUCE_TPB={REDUCE_TPB} exceeds MAX_THREADS_PER_BLOCK={device.MAX_THREADS_PER_BLOCK}")
        if REDUCE_TPB < 1 or (REDUCE_TPB & (REDUCE_TPB - 1)) != 0:
            raise ValueError("REDUCE_TPB must be a positive power of two")
        if REDUCE_BLOCKS_PER_SM < 1:
            raise ValueError("REDUCE_BLOCKS_PER_SM must be positive")
        reduce_shared_bytes = (2 * REDUCE_TPB * np.dtype(np.float64).itemsize)
        if reduce_shared_bytes > device.MAX_SHARED_MEMORY_PER_BLOCK:
            raise ValueError(
                f"Reduction requires {reduce_shared_bytes} bytes "
                f"of shared memory per block, but the device supports "
                f"only {device.MAX_SHARED_MEMORY_PER_BLOCK} bytes"
            )

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
    def _validate_run_parameters(max_iters, tol, check_every, fixed_iters,):
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

    def _owned_update_row_bounds(self):
        first = 1
        last_exclusive = self.local_ny + 1
        if self.ystart == 0:
            first = 2
        if self.yend == self.ny:
            last_exclusive = self.local_ny
        return first, last_exclusive

    def _apply_dirichlet_bc_host(self, arr):
        with nvtx_range("_apply_dirichlet_bc_host", id_color=1):
            arr[1 : self.local_ny + 1, 0] = self.bc_left
            arr[1 : self.local_ny + 1, self.nx - 1] = self.bc_right
            if self.rank == 0:
                arr[1, :] = self.bc_bottom
            if self.rank == self.size - 1:
                arr[self.local_ny, :] = self.bc_top

    def exchange_halos_mpi(self):
        with nvtx_range("exchange_halos_mpi", id_color=2):
            cuda.default_stream().synchronize()
            self.mpi_comm.Sendrecv(
                sendbuf=self.d_u[1, :],
                dest=self.down,
                sendtag=100,
                recvbuf=self.d_u[0, :],
                source=self.down,
                recvtag=200,
            )
            self.mpi_comm.Sendrecv(
                sendbuf=self.d_u[self.local_ny, :],
                dest=self.up,
                sendtag=200,
                recvbuf=self.d_u[self.local_ny + 1, :],
                source=self.up,
                recvtag=100,
            )

    def jacobi_step(self):
        if self.jacobi_blocks is None:
            return
        with nvtx_range("jacobi_step", id_color=3):
            jacobi_kernel[self.jacobi_blocks, self.jacobi_threads](
                self.d_u, self.d_unew, self.nx, self.update_i0, self.update_i1, self.dx2, self.dy2, self.inv_denom,
            )

    def compute_relative_l2_update_norm(self):
        with nvtx_range("jacobi_l2", id_color=4):
            if self.local_update_count > 0:
                shared_reduce_norm_kernel[self.norm_blocks, REDUCE_TPB](
                    self.d_u,
                    self.d_unew,
                    self.nx,
                    self.update_i0,
                    self.update_i1,
                    self.d_partial_diff,
                    self.d_partial_norm,
                )
                # Second reduction pass over the small number of blocks using existing @cuda.reduce
                local_diff2 = sum_reduce(self.d_partial_diff, size=self.norm_blocks)
                local_norm2 = sum_reduce(self.d_partial_norm, size=self.norm_blocks)
            else:
                local_diff2 = 0.0
                local_norm2 = 0.0
                
            self.h_partial[0] = float(local_diff2)
            self.h_partial[1] = float(local_norm2)
            self.mpi_comm.Allreduce(self.h_partial, self.h_global, op=MPI.SUM)
            diff2 = self.h_global[0]
            norm2 = self.h_global[1]
            return float(np.sqrt(diff2) / max(np.sqrt(norm2), 1.0e-300))

    def reset_fields(self):
        self.d_u.copy_to_device(self.h_initial)
        self.d_unew.copy_to_device(self.h_initial)
        cuda.synchronize()
        self.mpi_comm.Barrier()

    def warmup(self):
        with nvtx_range("warmup", id_color=5):
            self.mpi_comm.Barrier()
            self.exchange_halos_mpi()
            self.jacobi_step()
            _ = self.compute_relative_l2_update_norm()
            cuda.synchronize()
            self.mpi_comm.Barrier()
        self.reset_fields()


    def run(self, max_iters, tol, check_every, fixed_iters=None):
        rel_update_norm = np.inf
        niters = 0
        with nvtx_range("jacobi_run", id_color=6):
            self._validate_run_parameters(max_iters, tol, check_every, fixed_iters)
            iteration_limit = fixed_iters if fixed_iters is not None else max_iters
            start_event = cuda.event()
            stop_event  = cuda.event()
            cuda.synchronize()
            self.mpi_comm.Barrier()
            wall_t0 = MPI.Wtime()
            start_event.record()
            for it in range(1, iteration_limit + 1):
                self.exchange_halos_mpi()
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
            stop_event.record()
            stop_event.synchronize()
            elapsed_cuda_local = (cuda.event_elapsed_time(start_event, stop_event,) * 1.0e-3)
            elapsed_wall_local = MPI.Wtime() - wall_t0
            elapsed_cuda_max = self.mpi_comm.allreduce(elapsed_cuda_local,op=MPI.MAX,)
            elapsed_wall_max = self.mpi_comm.allreduce(elapsed_wall_local,op=MPI.MAX,)
        return niters, rel_update_norm, elapsed_cuda_max, elapsed_wall_max

    def write_parallel_hdf5(self, filename, niters=None, rel_update_norm=None):
        with nvtx_range("jacobi_write", id_color=7):
            import h5py 
            if not h5py.get_config().mpi:
                raise RuntimeError(
                    "This h5py build does not have MPI support. "
                    "Rebuild h5py against parallel HDF5 with HDF5_MPI=ON."
                )
            # Ensure parent directory exists across rank 0 (or all ranks safely)
            if self.rank == 0:
                out_dir = os.path.dirname(filename)
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
            self.mpi_comm.Barrier()
            local_host = self.d_u[1:self.local_ny + 1,:].copy_to_host()
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
                dset.attrs["decomposition"] = "1D row decomposition"
                dset.attrs["backend"] = ("Numba CUDA + CUDA-aware MPI halo exchange + mpi4py")
                dset.attrs["communication"] = ("CUDA-aware MPI Sendrecv halos; host MPI Allreduce for norm")
                dset.attrs["rel_update_norm_method"] = ("Numba CUDA block-shared partial reduction + cuda.reduce + host MPI Allreduce")
                dset.attrs["convergence_metric"] = ("relative L2 iterate-update norm over global physical interior")
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
    parser.add_argument("--output", type=str, default=None, help="Optional parallel HDF5 output filename.",)
    parser.add_argument("--fixed-iters", type=int, default=None,
        help=("Run exactly this many iterations without tolerance-based termination. "
              "The update norm is still evaluated according to --check-every."),
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
    
    if size > args.ny:
        if rank == 0:
            print("Error: number of MPI ranks must be <= ny for full-row decomposition.", flush=True,)
        mpi_comm.Abort(1)
    if rank == 0 and size > args.ny - 2:
        print("Warning: one or more MPI ranks own only physical boundary rows and perform no Jacobi updates.", flush=True,)
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
    
    provided_thread_level = MPI.Query_thread()
    if provided_thread_level < MPI.THREAD_FUNNELED:
        if rank == 0:
            print("Error: MPI did not provide MPI_THREAD_FUNNELED.", flush=True,)
        mpi_comm.Abort(1)

    try:
        local_rank = get_local_rank(mpi_comm)
    except Exception as exc:
        print(f"rank {rank}: failed to determine local rank: {exc}", flush=True,)
        mpi_comm.Abort(1)

    try:
        ndev = len(cuda.gpus)
    except Exception as exc:
        print(f"rank {rank}: failed to query CUDA devices: {exc}", flush=True,)
        mpi_comm.Abort(1)

    if ndev <= 0:
        print(f"rank {rank}: no visible CUDA devices", flush=True,)
        mpi_comm.Abort(1)

    device_id = 0 if ndev == 1 else local_rank
    if device_id >= ndev:
        print(f"rank {rank}: cannot map local_rank={local_rank} to {ndev} visible GPU(s)", flush=True,)
        mpi_comm.Abort(1)
    
    try:
        cuda.select_device(device_id)
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
    except Exception as exc:
        print(f"rank {rank}: solver initialization failed: {exc}", flush=True,)
        mpi_comm.Abort(1)

    if rank == 0:
        print("Distributed Numba CUDA / CUDA-aware MPI Laplace solver")
        print(f"  global grid        : ny={args.ny}, nx={args.nx}")
        print(f"  physical size      : ly={args.ly}, lx={args.lx}")
        print(f"  MPI ranks          : {size}")
        print(f"  halo mode          : CUDA-aware MPI")
        print(f"  check every        : {args.check_every}")
        print(f"  output             : {args.output}")
        if args.fixed_iters is not None:
            print(f"  mode               : fixed iteration")
            print(f"  fixed iterations   : {args.fixed_iters}")
        else:
            print(f"  mode               : tolerance")
            print(f"  tolerance          : {args.tol}")
            print(f"  max iterations     : {args.max_iters}")
        print("", flush=True)
    
    try:
        device = cuda.current_context().device
        gpu_name = (device.name.decode() if isinstance(device.name, bytes) else str(device.name))
    except Exception as exc:
        print(f"rank {rank}: failed to query CUDA device properties: {exc}", flush=True)
        mpi_comm.Abort(1)

    itemsize = np.dtype(np.float64).itemsize
    local_field_bytes = ((solver.local_ny + 2) * solver.nx * itemsize)
    norm_workspace_bytes  = (2 * solver.norm_workspace_size * itemsize)
    explicit_device_bytes = (2 * local_field_bytes + norm_workspace_bytes)
    
    ordered_print(
        mpi_comm,
        (
            f"rank {rank:4d}/{size} | "
            f"local_rank={local_rank:2d} | "
            f"cuda_device={device_id:2d} | "
            f"gpu={gpu_name} | "
            f"SMs={solver.num_sms} | "
            f"global rows [{solver.ystart}:{solver.yend}) | "
            f"local_ny={solver.local_ny} | "
            f"local array={solver.d_u.shape} | "
            f"local_updates={solver.local_update_count} | "
            f"update_rows={solver.update_i1 - solver.update_i0} | "
            f"reduce_tpb={REDUCE_TPB} | "
            f"reduce_blocks={solver.norm_blocks} | "
            f"reduce_blocks_per_sm={REDUCE_BLOCKS_PER_SM} | "
            f"memory/field={local_field_bytes / 1024**2:.2f} MiB | "
            f"norm_workspace={norm_workspace_bytes / 1024:.2f} KiB | "
            f"explicit_device={explicit_device_bytes / 1024**2:.2f} MiB"
        ),
    )

    global_bytes_per_array = (args.ny * args.nx * np.dtype(np.float64).itemsize)
    if rank == 0:
        print("", flush=True)
        print(f"  global field size : {global_bytes_per_array / 1024**2:.2f} MiB")
        print("", flush=True)

    if rank == 0:
        print("Warming up Numba CUDA + MPI...", flush=True)
    
    try:
        solver.warmup()
    except Exception as exc:
        print(f"rank {rank}: CUDA-aware MPI warm-up failed: {exc}", flush=True,)
        mpi_comm.Abort(1)

    if rank == 0:
        print("Warm-up complete.", flush=True)
    
    try:
        niters, rel_update_norm, elapsed_cuda_event, elapsed_wall = solver.run(
            max_iters=args.max_iters,
            tol=args.tol,
            check_every=args.check_every,
            fixed_iters=args.fixed_iters,
        )
    except Exception as exc:
        print(f"rank {rank}: solver run failed: {exc}", flush=True,)
        mpi_comm.Abort(1)

    converged = (args.fixed_iters is None and rel_update_norm < args.tol)
    if rank == 0 and args.fixed_iters is None and not converged:
        print("Warning: maximum iteration count reached before the requested tolerance was satisfied.", flush=True,)
    global_updates_per_iter = ((args.nx - 2) * (args.ny - 2))
    total_updates = (global_updates_per_iter * niters)
    gupdates_per_second_cuda_event = (total_updates / elapsed_cuda_event / 1.0e9 if elapsed_cuda_event > 0.0 else float("inf"))
    gupdates_per_second_wall       = (total_updates / elapsed_wall       / 1.0e9 if elapsed_wall       > 0.0 else float("inf"))


    if rank == 0:
        print()
        print("============================================================")
        print("Solver summary")
        print("============================================================")
        print("Implementation                                             : Numba CUDA fused Jacobi + CUDA-aware MPI")
        print(f"Numba CUDA + MPI iterations                              : {niters}")
        if args.fixed_iters is None:
            print(f"Numba CUDA + MPI converged                               : {converged}")
        else:
            print("Numba CUDA + MPI convergence used to stop                : no")

        print(f"Numba CUDA + MPI final relative iterate-update norm      : {rel_update_norm:.6e}")
        print(f"Numba CUDA + MPI CUDA-event interval [s]                 : {elapsed_cuda_event:.6f}")
        print(f"Numba CUDA + MPI wall elapsed time [s]                   : {elapsed_wall:.6f}")
        print(f"Numba CUDA + MPI CUDA-event time/iter [ms]               : {1000.0 * elapsed_cuda_event / max(niters, 1):.6f}")
        print(f"Numba CUDA + MPI wall time/iter [ms]                     : {1000.0 * elapsed_wall / max(niters, 1):.6f}")
        print("Throughput metric: effective monitored distributed solver")
        print(f"Numba CUDA + MPI CUDA-event throughput [GCell-updates/s] : {gupdates_per_second_cuda_event:.6f}")
        print(f"Numba CUDA + MPI wall throughput [GCell-updates/s]       : {gupdates_per_second_wall:.6f}")
    if args.output is not None:
        try:
            solver.write_parallel_hdf5(args.output, niters, rel_update_norm,)
        except Exception as exc:
            print(f"rank {rank}: parallel HDF5 output failed: {exc}", flush=True,)
            mpi_comm.Abort(1)
        if rank == 0:
            print(f"Numba CUDA + MPI wrote solution                          : {args.output}")
    if rank == 0:
        print("", flush=True)


if __name__ == "__main__":
    main()
