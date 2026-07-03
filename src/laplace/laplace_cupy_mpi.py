#!/usr/bin/env python3
import os
import argparse
import mpi4py
mpi4py.rc.thread_level = "funneled"
from mpi4py import MPI
import numpy as np
import cupy as cp

# This example requires CUDA-aware MPI.
# If the MPI library is not CUDA-aware,
# passing CuPy arrays directly to Sendrecv/Allreduce
# may fail or silently stage through host memory depending on the implementation.

# RawKernel assumes contiguous row-major arrays.
# Do not pass transposed or strided views unless the kernel explicitly handles strides.

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
        self.rel_update_norm_threads = 256
        if self.rel_update_norm_threads & (self.rel_update_norm_threads - 1) != 0:
            raise ValueError("rel_update_norm_threads must be a power of two")
        max_local_interior = self.local_ny * (self.nx - 2)
        self.rel_update_norm_blocks = min(4096, (max_local_interior + self.rel_update_norm_threads - 1) // self.rel_update_norm_threads,)
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
        cp.cuda.nvtx.RangePush("_owned_update_row_bounds")
        first = 1
        last_exclusive = self.local_ny + 1
        try:
            if self.ystart == 0:
                first = 2
            if self.yend == self.ny:
                last_exclusive = self.local_ny
        finally:
            cp.cuda.nvtx.RangePop()
        return first, last_exclusive

    def _apply_dirichlet_bc(self, arr):
        """
        Apply physical Dirichlet boundary conditions.
        """
        cp.cuda.nvtx.RangePush("_apply_dirichlet_bc")
        try:
            arr[1 : self.local_ny + 1, 0] = self.bc_left
            arr[1 : self.local_ny + 1, self.nx - 1] = self.bc_right
            if self.rank == 0:
                arr[1, :] = self.bc_bottom
            if self.rank == self.size - 1:
                arr[self.local_ny, :] = self.bc_top
        finally:
            cp.cuda.nvtx.RangePop()

    def exchange_halos_mpi(self):
        """
        Exchange halo rows using CUDA-aware MPI Sendrecv.
        Tags:
          - 100: upward send, rank r sends row 1 to rank r-1
          - 200: downward send, rank r sends row local_ny to rank r+1
        """
        cp.cuda.nvtx.RangePush("exchange_halos_mpi")
        try:
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
        finally:
            cp.cuda.nvtx.RangePop()

    def jacobi_step(self):
        """
        One RawKernel Jacobi update.
        """
        cp.cuda.nvtx.RangePush("jacobi_step")
        try:
            i0, i1 = self._owned_update_row_bounds()
            if i0 >= i1:
                return
            threads = (32, 8)
            blocks = ((self.nx - 2 + threads[0] - 1) // threads[0], (i1 - i0 + threads[1] - 1) // threads[1],)
            jacobi_kernel(blocks, threads,
                (self.u, self.unew, np.uint64(self.nx), np.uint64(i0), np.uint64(i1), self.dx2, self.dy2, self.inv_denom,),)
        finally:
            cp.cuda.nvtx.RangePop()


    def compute_local_reduction_rawkernel(self, i0, i1):
        n = max(0, (i1 - i0) * (self.nx - 2))
        shared_mem_bytes = (2 * self.rel_update_norm_threads * cp.dtype(cp.float64).itemsize)
        rel_update_norm_partial_kernel((self.rel_update_norm_blocks,), (self.rel_update_norm_threads,),
            (
                self.u,
                self.unew,
                self.partial_diff,
                self.partial_norm,
                np.uint64(self.nx),
                np.uint64(i0),
                np.uint64(n),
            ),
            shared_mem=shared_mem_bytes,
        )
        local_diff2 = cp.sum(self.partial_diff)
        local_norm2 = cp.sum(self.partial_norm)
        return local_diff2, local_norm2


    def compute_relative_l2_update_norm(self):
        """
        Compute global relative L2 update norm:
            ||u_new - u||_2 / ||u_new||_2
        Local part uses the same RawKernel partial-reduction scheme
        as the serial CuPy RawKernel implementation.
        Global reduction uses CUDA-aware MPI Allreduce on a 2-element CuPy buffer.
        """
        cp.cuda.nvtx.RangePush("compute_relative_l2_update_norm")
        rel_update_norm = 0.0
        try:
            i0, i1 = self._owned_update_row_bounds()
            local_diff2, local_norm2 = self.compute_local_reduction_rawkernel(i0, i1)
            self.red_local[0] = local_diff2
            self.red_local[1] = local_norm2
            # Required before CUDA-aware MPI reads red_local.
            cp.cuda.get_current_stream().synchronize()
            self.mpi_comm.Allreduce(self.red_local, self.red_global, op=MPI.SUM,)
            diff2 = float(self.red_global[0].get())
            norm2 = float(self.red_global[1].get())
            rel_update_norm = np.sqrt(diff2) / max(np.sqrt(norm2), 1.0e-300)
        finally:
            cp.cuda.nvtx.RangePop()
        return float(rel_update_norm)

    def run(self, max_iters, tol, check_every, fixed_iters=None):
        """
        Run Jacobi iteration.
        """
        cp.cuda.nvtx.RangePush("run")
        rel_update_norm = np.inf
        niters = 0
        elapsed = 0.0
        try:
            Laplace2DDistributedCUDAAwareMPI._validate_run_parameters(max_iters, tol, check_every, fixed_iters)
            nstop = fixed_iters if fixed_iters is not None else max_iters
            self.mpi_comm.Barrier()
            start = cp.cuda.Event()
            stop = cp.cuda.Event()
            cp.cuda.get_current_stream().synchronize()
            start.record()
            for it in range(1, nstop + 1):
                self.exchange_halos_mpi()
                self.jacobi_step()
                do_check = (it == 1 or it % check_every == 0 or it == nstop)
                if do_check:
                    rel_update_norm = self.compute_relative_l2_update_norm()
                    if fixed_iters is None and rel_update_norm < tol:
                        # Keep the converged field in self.u for output.
                        self.u, self.unew = self.unew, self.u
                        niters = it
                        break
                self.u, self.unew = self.unew, self.u
                niters = it
            stop.record()
            stop.synchronize()
            self.mpi_comm.Barrier()
            elapsed_local = cp.cuda.get_elapsed_time(start, stop) / 1000.0
            elapsed = self.mpi_comm.allreduce(elapsed_local, op=MPI.MAX)
        finally:
            cp.cuda.nvtx.RangePop()
        return niters, rel_update_norm, elapsed

    def write_parallel_hdf5(self, filename, niters=None, rel_update_norm=None):
        """
        Write global solution using parallel HDF5.
        Note:
            HDF5 write uses host memory. We explicitly copy local GPU slab
            to NumPy before entering HDF5.
        """
        cp.cuda.nvtx.RangePush("write_parallel_hdf5")
        try:
            import h5py
            if not h5py.get_config().mpi:
                raise RuntimeError(
                    "This h5py build does not have MPI support. "
                    "Rebuild h5py against parallel HDF5 with HDF5_MPI=ON."
                )
            self.mpi_comm.Barrier()
            cp.cuda.get_current_stream().synchronize()
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
        finally:
            cp.cuda.nvtx.RangePop()

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
    parser.add_argument("--output", type=str, default="laplace.h5")
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
    ndev = cp.cuda.runtime.getDeviceCount()
    if ndev <= 0:
        raise RuntimeError("No CUDA devices found")
    if local_rank >= ndev:
        if rank == 0:
            print("Error: more local MPI ranks than visible CUDA devices.\n"
                  "Use one MPI rank per GPU or explicitly allow oversubscription.", 
                  flush=True,
                 )
        mpi_comm.Abort(1)
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
    local_bytes_per_array  = ((solver.local_ny + 2) * solver.nx * cp.dtype(cp.float64).itemsize)
    global_bytes_per_array = (args.ny * args.nx * cp.dtype(cp.float64).itemsize)
    ordered_print(mpi_comm,
        (
            f"rank {rank:4d}/{size} | "
            f"local_rank={local_rank:2d} | "
            f"cuda_device={device_id:2d} | "
            f"gpu={gpu_name} | "
            f"global rows [{solver.ystart}:{solver.yend}) | "
            f"local_ny={solver.local_ny} | "
            f"local array={solver.u.shape} | "
            f"memory/field={local_bytes_per_array / 1024**2:.2f} MiB"
        ),
    )
    if rank == 0:
        print("", flush=True)
        print(f"  global field size : {global_bytes_per_array / 1024**2:.2f} MiB")
        print("", flush=True)
    niters, rel_update_norm, elapsed = solver.run(
        max_iters=args.max_iters,
        tol=args.tol,
        check_every=args.check_every,
        fixed_iters=args.fixed_iters,
    )
    cp.cuda.get_current_stream().synchronize()
    solver.write_parallel_hdf5(args.output, niters, rel_update_norm)
    if rank == 0:
        print("", flush=True)
        print(f"iterations                  : {niters}")
        print(f"final relative update norm  : {rel_update_norm:.6e}")
        print(f"CuPy+MPI elapsed time [s]   : {elapsed:.6f}")
        print(f"CuPy+MPI time per iter [ms] : {1000.0 * elapsed / max(niters, 1):.6f}")
        print(f"wrote solution              : {args.output}")
        print("", flush=True)


if __name__ == "__main__":
    main()


