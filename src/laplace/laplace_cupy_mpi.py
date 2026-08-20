#!/usr/bin/env python3

import argparse
import os
from contextlib import contextmanager

import mpi4py

mpi4py.rc.thread_level = "funneled"

from mpi4py import MPI

import cupy as cp
import numpy as np

# This example requires CUDA-aware MPI.
# If the MPI library is not CUDA-aware,
# passing CuPy arrays directly to Sendrecv/Allreduce
# may fail or silently stage through host memory depending on the implementation.

# RawKernel assumes contiguous row-major arrays.
# Do not pass transposed or strided views unless the kernel explicitly handles strides.

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


jacobi_kernel = cp.RawKernel(r'''
extern "C" __global__
void jacobi_2d(
    const double* __restrict__ u,
    double* __restrict__ unew,
    const size_t nx,
    const size_t i0,
    const size_t i1,
    const double dx2,
    const double dy2,
    const double inv_denom
)
{
    size_t j = blockDim.x * blockIdx.x + threadIdx.x;
    size_t i = blockDim.y * blockIdx.y + threadIdx.y;
    j += 1;
    i += i0;
    if (j >= nx - 1 || i >= i1) {
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
void rel_update_norm_partial_local_2d(
    const double* __restrict__ u,
    const double* __restrict__ unew,
    double* __restrict__ partial_diff,
    double* __restrict__ partial_norm,
    const size_t nx,
    const size_t i0,
    const size_t n
)
{
    extern __shared__ double shared[];
    double* sdiff = shared;
    double* snorm = shared + blockDim.x;
    const unsigned int tid = threadIdx.x;
    const size_t stride = (size_t)blockDim.x * (size_t)gridDim.x;
    const size_t ncols = nx - 2;
    double diff_sum = 0.0;
    double norm_sum = 0.0;
    for (size_t k = (size_t)blockIdx.x * blockDim.x + tid; k < n; k += stride)
    {
        const size_t local_i = k / ncols;
        const size_t local_j = k - local_i * ncols;
        const size_t i = i0 + local_i;
        const size_t j = 1 + local_j;
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
''', 'rel_update_norm_partial_local_2d')


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


class Laplace2DDistributedCUDAAwareMPI:
    """
    Distributed Jacobi solver for the 2D Laplace equation.
    Decomposition:
        1D row decomposition along global y direction.
    Local layout:
        u[0, :]                 upper halo
        u[1:local_ny+1, :]      owned physical rows
        u[local_ny+1, :]        lower halo
    Communication:
        CUDA-aware MPI Sendrecv for halos.
        CUDA-aware MPI Allreduce for rel_update_norm.
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
        dtype=cp.float64,
    ):
        Laplace2DDistributedCUDAAwareMPI._validate_problem_parameters(nx, ny, lx, ly, bc_left, bc_right, bc_bottom, bc_top, dtype)
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
        self.dtype = dtype
        self.dx = self.lx / (self.nx - 1)
        self.dy = self.ly / (self.ny - 1)
        self.dx2 = self.dx * self.dx
        self.dy2 = self.dy * self.dy
        self.inv_denom = 1.0 / (2.0 * (self.dx2 + self.dy2))
        self.bc_left = float(bc_left)
        self.bc_right = float(bc_right)
        self.bc_bottom = float(bc_bottom)
        self.bc_top = float(bc_top)
        self.u = cp.zeros((self.local_ny + 2, self.nx), dtype=self.dtype)
        self.unew = cp.zeros_like(self.u)
        self._apply_dirichlet_bc(self.u)
        self._apply_dirichlet_bc(self.unew)
        # red[0] = global squared update difference
        # red[1] = global squared solution norm
        self.red_local = cp.zeros(2, dtype=cp.float64)
        self.red_global = cp.zeros(2, dtype=cp.float64)
        self.update_i0, self.update_i1 = self._owned_update_row_bounds()
        self.local_update_count = (max(0, self.update_i1 - self.update_i0) * (self.nx - 2))
        self.rel_update_norm_threads = 256
        if self.local_update_count > 0:
            self.rel_update_norm_blocks = min(4096,(self.local_update_count + self.rel_update_norm_threads - 1) // self.rel_update_norm_threads,)
        else:
            self.rel_update_norm_blocks = 1

        if self.rel_update_norm_threads & (self.rel_update_norm_threads - 1) != 0:
            raise ValueError("rel_update_norm_threads must be a power of two")

        self.jacobi_threads = (32, 8)
        if self.update_i0 < self.update_i1:
            self.jacobi_blocks = ( 
                (self.nx - 2 + self.jacobi_threads[0] - 1) // self.jacobi_threads[0],
                (self.update_i1 - self.update_i0 + self.jacobi_threads[1] - 1) // self.jacobi_threads[1],
            )
        else:
            self.jacobi_blocks = None
        
        self.rel_update_norm_shared_bytes = (2 * self.rel_update_norm_threads * cp.dtype(cp.float64).itemsize)
        self.local_update_count_kernel = np.uint64(self.local_update_count)

        self.nx_kernel = np.uint64(self.nx)
        self.update_i0_kernel = np.uint64(self.update_i0)
        self.update_i1_kernel = np.uint64(self.update_i1)
        self.dx2_kernel = np.float64(self.dx2)
        self.dy2_kernel = np.float64(self.dy2)
        self.inv_denom_kernel = np.float64(self.inv_denom)
        self._validate_kernel_configuration()
        self.partial_diff = cp.empty(self.rel_update_norm_blocks, dtype=cp.float64)
        self.partial_norm = cp.empty(self.rel_update_norm_blocks, dtype=cp.float64)

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
            raise ValueError(f"Reduction requires {self.rel_update_norm_shared_bytes} bytes of shared memory. "
                             f"But the device supports only {attributes['MaxSharedMemoryPerBlock']} bytes")

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
        """
        Return local row bounds for true update/rel_update_norm region.
        Excludes:
          - halo rows
          - global bottom boundary
          - global top boundary
        """
        first = 1
        last_exclusive = self.local_ny + 1
        with nvtx_range("_owned_update_row_bounds"):
            if self.ystart == 0:
                first = 2
            if self.yend == self.ny:
                last_exclusive = self.local_ny
        return first, last_exclusive

    def _apply_dirichlet_bc(self, arr):
        """
        Apply physical Dirichlet boundary conditions.
        """
        with nvtx_range("_apply_dirichlet_bc"):
            arr[1 : self.local_ny + 1, 0] = self.bc_left
            arr[1 : self.local_ny + 1, self.nx - 1] = self.bc_right
            if self.rank == 0:
                arr[1, :] = self.bc_bottom
            if self.rank == self.size - 1:
                arr[self.local_ny, :] = self.bc_top

    def exchange_halos_mpi(self):
        """
        Exchange halo rows using CUDA-aware MPI Sendrecv.
        Tags:
          - 100: upward send, rank r sends row 1 to rank r-1
          - 200: downward send, rank r sends row local_ny to rank r+1
        """
        with nvtx_range("exchange_halos_mpi"):
            stream = cp.cuda.get_current_stream()
            # Required for portable CUDA-aware MPI behavior:
            # make sure previous GPU writes to self.u are visible to MPI.
            stream.synchronize()
            # Exchange with upper neighbor:
            # rank r sends owned first row to r-1
            # rank r receives upper halo from r-1
            self.mpi_comm.Sendrecv(
                sendbuf=self.u[1, :],
                dest=self.up,
                sendtag=100,
                recvbuf=self.u[0, :],
                source=self.up,
                recvtag=200,
            )
            # Exchange with lower neighbor:
            # rank r sends owned last row to r+1
            # rank r receives lower halo from r+1
            self.mpi_comm.Sendrecv(
                sendbuf=self.u[self.local_ny, :],
                dest=self.down,
                sendtag=200,
                recvbuf=self.u[self.local_ny + 1, :],
                source=self.down,
                recvtag=100,
            )

    def jacobi_step(self):
        """
        One RawKernel Jacobi update.
        """
        if self.jacobi_blocks is None:
            return
        with nvtx_range("jacobi_step"):
            jacobi_kernel(
                self.jacobi_blocks, self.jacobi_threads,
                (
                    self.u, self.unew, self.nx_kernel, 
                    self.update_i0_kernel, self.update_i1_kernel, 
                    self.dx2_kernel, self.dy2_kernel, self.inv_denom_kernel,
                ),
            )

    def compute_local_reduction_rawkernel(self):
        if self.local_update_count == 0:
            return cp.asarray(0.0, dtype=cp.float64), cp.asarray(0.0, dtype=cp.float64)
        rel_update_norm_partial_kernel((self.rel_update_norm_blocks,), (self.rel_update_norm_threads,),
            (
                self.u,
                self.unew,
                self.partial_diff,
                self.partial_norm,
                self.nx_kernel,
                self.update_i0_kernel,
                self.local_update_count_kernel,
            ),
            shared_mem=self.rel_update_norm_shared_bytes,
        )
        local_diff2 = cp.sum(self.partial_diff[:self.rel_update_norm_blocks])
        local_norm2 = cp.sum(self.partial_norm[:self.rel_update_norm_blocks])
        return local_diff2, local_norm2


    def compute_relative_l2_update_norm(self):
        """
        Compute global relative L2 update norm:
            ||u_new - u||_2 / ||u_new||_2
        Local part uses the same RawKernel partial-reduction scheme
        as the serial CuPy RawKernel implementation.
        Global reduction uses CUDA-aware MPI Allreduce on a 2-element CuPy buffer.
        """
        with nvtx_range("compute_relative_l2_update_norm"):
            local_diff2, local_norm2 = self.compute_local_reduction_rawkernel()
            self.red_local[0] = local_diff2
            self.red_local[1] = local_norm2
            # Required before CUDA-aware MPI reads red_local.
            cp.cuda.get_current_stream().synchronize()
            self.mpi_comm.Allreduce(self.red_local, self.red_global, op=MPI.SUM,)
            diff2 = float(self.red_global[0].get())
            norm2 = float(self.red_global[1].get())
            return float(np.sqrt(diff2) / max(np.sqrt(norm2), 1.0e-300))


    def reset_fields(self):
        self.u.fill(0.0)
        self.unew.fill(0.0)
        self._apply_dirichlet_bc(self.u)
        self._apply_dirichlet_bc(self.unew)
        cp.cuda.get_current_stream().synchronize()
        self.mpi_comm.Barrier()


    def warmup(self):
        """
        Warm up RawKernel launch, reduction, and CUDA-aware MPI.
        """
        with nvtx_range("warmup"):
            self.mpi_comm.Barrier()
            self.exchange_halos_mpi()
            self.jacobi_step()
            _ = self.compute_relative_l2_update_norm()
            cp.cuda.get_current_stream().synchronize()
            self.mpi_comm.Barrier()
        self.reset_fields()


    def run(self, max_iters, tol, check_every, fixed_iters=None):
        """
        Run Jacobi iteration.
        """
        rel_update_norm = np.inf
        niters = 0
        with nvtx_range("run"):
            self._validate_run_parameters(max_iters, tol, check_every, fixed_iters)
            iteration_limit = fixed_iters if fixed_iters is not None else max_iters
            start = cp.cuda.Event()
            stop = cp.cuda.Event()
            cp.cuda.get_current_stream().synchronize()
            self.mpi_comm.Barrier()
            wall_t0 = MPI.Wtime()
            start.record()
            for it in range(1, iteration_limit + 1):
                self.exchange_halos_mpi()
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
            elapsed_cuda_local = cp.cuda.get_elapsed_time(start, stop) / 1000.0
            self.mpi_comm.Barrier()
            elapsed_wall_local = MPI.Wtime() - wall_t0
            elapsed_cuda_max = self.mpi_comm.allreduce(elapsed_cuda_local, op=MPI.MAX)
            elapsed_wall_max = self.mpi_comm.allreduce(elapsed_wall_local, op=MPI.MAX)
        return niters, rel_update_norm, elapsed_cuda_max, elapsed_wall_max

    def write_parallel_hdf5(self, filename, niters=None, rel_update_norm=None):
        """
        Write global solution using parallel HDF5.
        Note:
            HDF5 write uses host memory. We explicitly copy local GPU slab
            to NumPy before entering HDF5.
        """
        with nvtx_range("write_parallel_hdf5"):
            import h5py
            if not h5py.get_config().mpi:
                raise RuntimeError(
                    "This h5py build does not have MPI support. "
                    "Rebuild h5py against parallel HDF5 with HDF5_MPI=ON."
                )
            self.mpi_comm.Barrier()
            local_host = cp.asnumpy(self.u[1 : self.local_ny + 1, :])
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
                dset.attrs["dtype"] = str(self.u.dtype)
                dset.attrs["mpi_size"] = self.size
                dset.attrs["rank_decomposition"] = "rows/y"
                dset.attrs["bc_left"] = self.bc_left
                dset.attrs["bc_right"] = self.bc_right
                dset.attrs["bc_bottom"] = self.bc_bottom
                dset.attrs["bc_top"] = self.bc_top
                dset.attrs["solver"] = "Jacobi"
                dset.attrs["backend"] = "CuPy RawKernel + CUDA-aware MPI + mpi4py"
                dset.attrs["decomposition"] = "1D row decomposition"
                dset.attrs["communication"] = "CUDA-aware MPI Sendrecv halos"
                if niters is not None:
                    dset.attrs["niters"] = niters
                if rel_update_norm is not None:
                    dset.attrs["rel_update_norm"] = rel_update_norm
            self.mpi_comm.Barrier()

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Distributed 2D Laplace solver using CuPy RawKernel, "
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
        ndev = cp.cuda.runtime.getDeviceCount()
    except Exception as exc:
        print(f"rank {rank}: failed to query visible CUDA devices: {exc}", flush=True,)
        mpi_comm.Abort(1)
    if ndev <= 0:
        print(f"rank {rank}: no visible CUDA devices", flush=True,)
        mpi_comm.Abort(1)

    if local_rank >= ndev:
        print(f"rank {rank}: local_rank={local_rank} is out of range; visible CUDA devices on this node: {ndev}. "
               "Use one MPI rank per GPU or explicitly allow oversubscription.", flush=True,)
        mpi_comm.Abort(1)

    try:
        cp.cuda.Device(local_rank).use()
        solver = Laplace2DDistributedCUDAAwareMPI(
            nx=args.nx,
            ny=args.ny,
            lx=args.lx,
            ly=args.ly,
            mpi_comm=mpi_comm,
            bc_left=args.bc_left,
            bc_right=args.bc_right,
            bc_bottom=args.bc_bottom,
            bc_top=args.bc_top,
            dtype=cp.float64,
        )
    except Exception as exc:
        print(f"rank {rank}: solver initialization failed: {exc}", flush=True,)
        mpi_comm.Abort(1)
    if rank == 0:
        print("Distributed CuPy RawKernel / CUDA-aware MPI Laplace solver")
        print(f"  global grid       : ny={args.ny}, nx={args.nx}")
        print(f"  physical size     : ly={args.ly}, lx={args.lx}")
        print(f"  MPI ranks         : {size}")
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
    device_id = cp.cuda.Device().id
    props = cp.cuda.runtime.getDeviceProperties(device_id)
    raw_name = props["name"]
    gpu_name = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)

    itemsize = np.dtype(np.float64).itemsize
    local_field_bytes = ((solver.local_ny + 2) * solver.nx * itemsize)
    partial_bytes = (2 * solver.rel_update_norm_blocks * itemsize)
    reduction_buffer_bytes = 4 * itemsize
    explicit_local_device_bytes = (2 * local_field_bytes + partial_bytes + reduction_buffer_bytes)
    global_bytes_per_array = (args.ny * args.nx * itemsize)

    ordered_print(mpi_comm,
        (
            f"rank {rank:4d}/{size} | "
            f"local_rank={local_rank:2d} | "
            f"cuda_device={device_id:2d} | "
            f"gpu={gpu_name} | "
            f"global rows [{solver.ystart}:{solver.yend}) | "
            f"local_ny={solver.local_ny} | "
            f"local array={solver.u.shape} | "
            f"memory/field={local_field_bytes / 1024**2:.2f} MiB | "
            f"explicit_device={explicit_local_device_bytes / 1024**2:.2f} MiB"
        ),
    )
    if rank == 0:
        print("", flush=True)
        print(f"  global field size : {global_bytes_per_array / 1024**2:.2f} MiB")
        print("", flush=True)

    if rank == 0:
        print("Warming up CuPy RawKernel + MPI...", flush=True)
    
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
    global_updates_per_iter = ((args.nx - 2) * (args.ny - 2))
    total_updates = global_updates_per_iter * niters
    gupdates_per_second_cuda_event = (total_updates / elapsed_cuda_event / 1.0e9 if elapsed_cuda_event > 0.0 else float("inf"))
    gupdates_per_second_wall       = (total_updates / elapsed_wall       / 1.0e9 if elapsed_wall       > 0.0 else float("inf"))

    if rank == 0:
        print()
        print("============================================================", flush=True)
        print("Solver summary", flush=True)
        print("============================================================", flush=True)
        print(f"CuPy RawKernel + MPI iterations                  : {niters}", flush=True)
        if args.fixed_iters is None:
            print(f"CuPy RawKernel + MPI converged                   : {converged}", flush=True,)
        else:
            print( "CuPy RawKernel + MPI convergence used            : no", flush=True,)

        print(f"CuPy RawKernel + MPI final relative update norm  : {rel_update_norm:.6e}", flush=True,)
        print(f"CuPy RawKernel + MPI CUDA-event interval [s]     : {elapsed_cuda_event:.6f}", flush=True)
        print(f"CuPy RawKernel + MPI wall elapsed time [s]       : {elapsed_wall:.6f}", flush=True)
        print(f"CuPy RawKernel + MPI CUDA-event time/iter [ms]   : {1000.0 * elapsed_cuda_event / max(niters, 1):.6f}", flush=True,)
        print(f"CuPy RawKernel + MPI wall time/iter [ms]         : {1000.0 * elapsed_wall / max(niters, 1):.6f}", flush=True,)
        print(f"CuPy RawKernel + MPI CUDA-event GUpdates/s       : {gupdates_per_second_cuda_event:.6f}", flush=True)
        print(f"CuPy RawKernel + MPI wall GUpdates/s             : {gupdates_per_second_wall:.6f}", flush=True)
    if args.output is not None:
        try:
            solver.write_parallel_hdf5(args.output, niters, rel_update_norm,)
        except Exception as exc:
            print(f"rank {rank}: parallel HDF5 output failed: {exc}", flush=True,)
            mpi_comm.Abort(1)
        if rank == 0:
            print(f"CuPy RawKernel + MPI wrote solution             : {args.output}", flush=True)
            print("", flush=True)


if __name__ == "__main__":
    main()


