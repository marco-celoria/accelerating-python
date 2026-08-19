#!/usr/bin/env python3
import argparse
import os
from contextlib import contextmanager

import numpy as np
import cupy as cp
import mpi4py

mpi4py.rc.thread_level = "funneled"
from mpi4py import MPI
from cupyx.distributed import init_process_group


# NCCL is used for GPU-GPU point-to-point communication and collectives.
# MPI is used for process launch, rank discovery, timing aggregation, and
# parallel HDF5.
#
# The RawKernels below assume contiguous C-order arrays. Do not pass
# transposed or otherwise strided views unless the kernels are updated to
# account for strides.


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


jacobi_kernel = cp.RawKernel(
    r'''
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

    j += 1;   // Skip the left physical boundary.
    i += i0;  // Shift to the requested local row interval.

    if (j >= nx - 1 || i >= i1)
        return;

    const size_t idx = i * nx + j;
    const double left  = u[idx - 1];
    const double right = u[idx + 1];
    const double up    = u[idx - nx];
    const double down  = u[idx + nx];

    unew[idx] =
        ((left + right) * dy2 + (up + down) * dx2) * inv_denom;
}
''',
    "jacobi_2d",
)


rel_update_norm_partial_kernel = cp.RawKernel(
    r'''
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

    for (size_t k = (size_t)blockIdx.x * blockDim.x + tid;
         k < n;
         k += stride)
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
''',
    "rel_update_norm_partial_local_2d",
)


def get_local_rank(mpi_comm):
    """Determine the GPU-local rank."""
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
    """Print one rank at a time in MPI rank order."""
    rank = mpi_comm.Get_rank()
    size = mpi_comm.Get_size()
    for r in range(size):
        mpi_comm.Barrier()
        if rank == r:
            print(message, flush=True)
    mpi_comm.Barrier()


def decompose_1d(n_global, size, rank):
    """Return a remainder-aware 1D block decomposition."""
    counts = np.full(size, n_global // size, dtype=np.int64)
    counts[: n_global % size] += 1

    offsets = np.zeros(size + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)

    local_n = int(counts[rank])
    start = int(offsets[rank])
    end = int(offsets[rank + 1])
    return local_n, start, end, counts, offsets


class Laplace2DDistributedNCCL:
    """
    Distributed Jacobi solver for the 2D Laplace equation.

    The global grid has ``ny`` rows and ``nx`` columns and is decomposed
    along y. Each rank stores one upper and one lower halo row:

        u[0, :]                 upper halo
        u[1:local_ny + 1, :]    owned physical rows
        u[local_ny + 1, :]      lower halo

    Communication/computation overlap:

      * ``comm_stream`` exchanges the top and bottom owned rows with NCCL.
      * ``compute_stream`` first updates rows that need no remote halo.
      * ``compute_stream`` then waits on ``halos_ready_event`` and updates
        the one interface row adjacent to each remote neighbor.

    ``u_ready_event`` protects the next iteration's NCCL reads from starting
    before the current field has been completely produced.
    """

    def __init__(
        self,
        nx,
        ny,
        lx,
        ly,
        nccl_comm,
        mpi_comm,
        bc_left=0.0,
        bc_right=1.0,
        bc_bottom=0.0,
        bc_top=0.0,
        dtype=cp.float64,
    ):
        self._validate_problem_parameters(
            nx,
            ny,
            lx,
            ly,
            bc_left,
            bc_right,
            bc_bottom,
            bc_top,
            dtype,
        )

        self.nx = int(nx)
        self.ny = int(ny)
        self.lx = float(lx)
        self.ly = float(ly)
        self.nccl_comm = nccl_comm
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
                "Each rank must own at least one row; use no more than ny ranks."
            )

        self.up = self.rank - 1 if self.rank > 0 else None
        self.down = self.rank + 1 if self.rank < self.size - 1 else None
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

        # Use non-blocking streams to avoid legacy-default-stream barriers.
        self.compute_stream = cp.cuda.Stream(non_blocking=True)
        self.comm_stream = cp.cuda.Stream(non_blocking=True)
        self.u_ready_event = cp.cuda.Event(disable_timing=True)
        self.halos_ready_event = cp.cuda.Event(disable_timing=True)

        # Allocate and initialize on the compute stream. The recorded event
        # makes this initialization visible to the communication stream.
        with self.compute_stream:
            self.u = cp.zeros(
                (self.local_ny + 2, self.nx), dtype=self.dtype
            )
            self.unew = cp.zeros_like(self.u)
            self._apply_dirichlet_bc(self.u)
            self._apply_dirichlet_bc(self.unew)

            self.red_local = cp.zeros(2, dtype=cp.float64)
            self.red_global = cp.zeros(2, dtype=cp.float64)

            self.rel_update_norm_threads = 256
            if (
                self.rel_update_norm_threads
                & (self.rel_update_norm_threads - 1)
            ) != 0:
                raise ValueError(
                    "rel_update_norm_threads must be a power of two"
                )

            max_local_interior = self.local_ny * (self.nx - 2)
            self.rel_update_norm_blocks = min(
                4096,
                (
                    max_local_interior
                    + self.rel_update_norm_threads
                    - 1
                )
                // self.rel_update_norm_threads,
            )
            self.partial_diff = cp.empty(
                self.rel_update_norm_blocks, dtype=cp.float64
            )
            self.partial_norm = cp.empty(
                self.rel_update_norm_blocks, dtype=cp.float64
            )

            self.u_ready_event.record(self.compute_stream)

    @staticmethod
    def _validate_problem_parameters(
        nx,
        ny,
        lx,
        ly,
        bc_left,
        bc_right,
        bc_bottom,
        bc_top,
        dtype,
    ):
        if not isinstance(nx, (int, np.integer)) or not isinstance(
            ny, (int, np.integer)
        ):
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
            raise ValueError(
                "This RawKernel implementation currently supports only cp.float64"
            )

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
                raise ValueError(
                    "tol must be finite and positive unless fixed_iters is used"
                )

    def _owned_update_row_bounds(self):
        """Return local half-open row bounds for the Jacobi update."""
        first = 1
        last_exclusive = self.local_ny + 1

        if self.ystart == 0:
            # Local row 1 is global row 0, a physical boundary.
            first = 2
        if self.yend == self.ny:
            # Local row local_ny is global row ny - 1, a physical boundary.
            last_exclusive = self.local_ny

        return first, last_exclusive

    def _apply_dirichlet_bc(self, arr):
        """Apply physical Dirichlet boundary conditions."""
        # Left and right boundaries exist on every rank.
        arr[1 : self.local_ny + 1, 0] = self.bc_left
        arr[1 : self.local_ny + 1, self.nx - 1] = self.bc_right

        # The y-boundary assignments intentionally take precedence at corners.
        if self.rank == 0:
            arr[1, :] = self.bc_bottom
        if self.rank == self.size - 1:
            arr[self.local_ny, :] = self.bc_top

    def exchange_halos_nccl(self, stream):
        """Enqueue nearest-neighbor halo exchange on ``stream``."""
        with nvtx_range("nccl_halo_exchange"):
            if self.up is not None:
                self.nccl_comm.send_recv(
                    in_array=self.u[1, :],
                    out_array=self.u[0, :],
                    peer=self.up,
                    stream=stream,
                )

            if self.down is not None:
                self.nccl_comm.send_recv(
                    in_array=self.u[self.local_ny, :],
                    out_array=self.u[self.local_ny + 1, :],
                    peer=self.down,
                    stream=stream,
                )

    def _launch_jacobi_rows(self, i0, i1, stream):
        """Enqueue Jacobi updates for local rows ``[i0, i1)``."""
        if i0 >= i1:
            return

        threads = (32, 8)
        blocks = (
            (self.nx - 2 + threads[0] - 1) // threads[0],
            (i1 - i0 + threads[1] - 1) // threads[1],
        )

        jacobi_kernel(
            blocks,
            threads,
            (
                self.u,
                self.unew,
                np.uint64(self.nx),
                np.uint64(i0),
                np.uint64(i1),
                self.dx2,
                self.dy2,
                self.inv_denom,
            ),
            stream=stream,
        )

    def _enqueue_overlapped_jacobi_step(self):
        """
        Enqueue one Jacobi step with communication/computation overlap.

        The deep interior does not depend on received halo rows. The first
        updated row needs the upper halo only when an upper rank exists, and
        the last updated row needs the lower halo only when a lower rank
        exists.
        """
        with nvtx_range("jacobi_step_overlapped"):
            i0, i1 = self._owned_update_row_bounds()

            # Every rank must participate in its matching NCCL operations,
            # including a rank that happens to own only a physical boundary
            # row and therefore has no Jacobi row to update. Returning before
            # the exchange would leave a neighbor's send_recv unmatched.
            #
            # NCCL reads boundary rows from self.u. Wait until the previous
            # iteration has completely produced that field.
            self.comm_stream.wait_event(self.u_ready_event)
            self.exchange_halos_nccl(self.comm_stream)
            self.halos_ready_event.record(self.comm_stream)

            if i0 >= i1:
                # Preserve NCCL ordering before a possible all-reduce and
                # before publishing the next u_ready_event.
                self.compute_stream.wait_event(self.halos_ready_event)
                return

            upper_interface = i0 if self.up is not None else None
            lower_interface = i1 - 1 if self.down is not None else None

            deep_i0 = i0 + int(self.up is not None)
            deep_i1 = i1 - int(self.down is not None)

            with nvtx_range("jacobi_deep_interior"):
                self._launch_jacobi_rows(
                    deep_i0, deep_i1, self.compute_stream
                )

            # Only halo-dependent kernels wait. Previously enqueued deep
            # interior work remains eligible to overlap with NCCL.
            self.compute_stream.wait_event(self.halos_ready_event)

            with nvtx_range("jacobi_interfaces"):
                if upper_interface is not None:
                    self._launch_jacobi_rows(
                        upper_interface,
                        upper_interface + 1,
                        self.compute_stream,
                    )

                if (
                    lower_interface is not None
                    and lower_interface != upper_interface
                ):
                    self._launch_jacobi_rows(
                        lower_interface,
                        lower_interface + 1,
                        self.compute_stream,
                    )

    def compute_local_reduction_rawkernel(self, i0, i1, stream):
        """Enqueue the local partial reductions on ``stream``."""
        n = max(0, (i1 - i0) * (self.nx - 2))
        if n == 0:
            # This path is not expected under the command-line rank limit,
            # but keeping it makes the method robust for boundary-only ranks.
            with stream:
                zero = cp.asarray(0.0, dtype=cp.float64)
            return zero, zero

        threads = self.rel_update_norm_threads
        blocks = min(4096, (n + threads - 1) // threads)
        shared_mem_bytes = 2 * threads * cp.dtype(cp.float64).itemsize

        rel_update_norm_partial_kernel(
            (blocks,),
            (threads,),
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
            stream=stream,
        )

        with stream:
            local_diff2 = cp.sum(self.partial_diff[:blocks])
            local_norm2 = cp.sum(self.partial_norm[:blocks])

        return local_diff2, local_norm2

    def compute_relative_l2_update_norm(self):
        """
        Compute ||unew - u||_2 / ||unew||_2 globally.

        This method is called after all Jacobi kernels have been enqueued on
        ``compute_stream``. Stream ordering therefore makes the reduction wait
        for the complete update without a device-wide synchronization.
        """
        with nvtx_range("compute_relative_l2_update_norm"):
            stream = self.compute_stream
            i0, i1 = self._owned_update_row_bounds()

            local_diff2, local_norm2 = self.compute_local_reduction_rawkernel(
                i0, i1, stream
            )

            with stream:
                self.red_local[0] = local_diff2
                self.red_local[1] = local_norm2

            self.nccl_comm.all_reduce(
                self.red_local,
                self.red_global,
                op="sum",
                stream=stream,
            )

            # The CPU needs the scalar convergence result, so synchronization
            # is unavoidable at check iterations. Copy both values together.
            stream.synchronize()
            diff2, norm2 = cp.asnumpy(self.red_global)

            rel_update_norm = np.sqrt(diff2) / max(
                np.sqrt(norm2), 1.0e-300
            )
            return float(rel_update_norm)

    def _finish_iteration(self):
        """Swap fields and publish the newly computed field to NCCL."""
        self.u, self.unew = self.unew, self.u

        # At this point the host has swapped references, but GPU work may still
        # be pending. Recording on compute_stream publishes completion of all
        # preceding kernels to the next iteration's communication stream.
        self.u_ready_event.record(self.compute_stream)

    def reset_fields(self):
        """Reset both fields and re-establish the initial stream dependency."""
        with nvtx_range("reset_fields"):
            with self.compute_stream:
                self.u.fill(0.0)
                self.unew.fill(0.0)
                self._apply_dirichlet_bc(self.u)
                self._apply_dirichlet_bc(self.unew)
                self.u_ready_event.record(self.compute_stream)

            self.compute_stream.synchronize()
            self.comm_stream.synchronize()
            self.mpi_comm.Barrier()

    def warmup(self):
        """Warm up RawKernels, NCCL halo exchange, and NCCL all-reduce."""
        with nvtx_range("warmup"):
            self.mpi_comm.Barrier()
            self._enqueue_overlapped_jacobi_step()
            _ = self.compute_relative_l2_update_norm()
            self._finish_iteration()
            self.compute_stream.synchronize()
            self.comm_stream.synchronize()
            self.mpi_comm.Barrier()

        self.reset_fields()

    def run(self, max_iters, tol, check_every, fixed_iters=None):
        """Run Jacobi iteration and return iterations, norm, and elapsed time."""
        self._validate_run_parameters(
            max_iters, tol, check_every, fixed_iters
        )

        rel_update_norm = np.inf
        niters = 0
        nstop = fixed_iters if fixed_iters is not None else max_iters

        with nvtx_range("run"):
            self.mpi_comm.Barrier()

            start = cp.cuda.Event()
            stop = cp.cuda.Event()

            # The initial field is already represented by u_ready_event.
            # Start timing on compute_stream and make comm_stream wait on the
            # same timing origin.
            start.record(self.compute_stream)
            self.comm_stream.wait_event(start)

            for it in range(1, nstop + 1):
                self._enqueue_overlapped_jacobi_step()

                do_check = (
                    it == 1
                    or it % check_every == 0
                    or it == nstop
                )

                if do_check:
                    rel_update_norm = (
                        self.compute_relative_l2_update_norm()
                    )

                # Always finish and swap exactly once per iteration. Thus
                # self.u always contains the latest completed/enqueued field.
                self._finish_iteration()
                niters = it

                if (
                    do_check
                    and fixed_iters is None
                    and rel_update_norm < tol
                ):
                    break

            # Every communication operation needed by the final update is an
            # ancestor of compute_stream through halos_ready_event.
            stop.record(self.compute_stream)
            stop.synchronize()
            self.comm_stream.synchronize()

            elapsed_local = (
                cp.cuda.get_elapsed_time(start, stop) / 1000.0
            )

            self.mpi_comm.Barrier()
            elapsed = self.mpi_comm.allreduce(
                elapsed_local, op=MPI.MAX
            )

        return niters, rel_update_norm, elapsed

    def write_parallel_hdf5(
        self, filename, niters=None, rel_update_norm=None
    ):
        """Write the distributed physical field through parallel HDF5."""
        with nvtx_range("write_parallel_hdf5"):
            import h5py

            if not h5py.get_config().mpi:
                raise RuntimeError(
                    "This h5py build does not have MPI support. Rebuild h5py "
                    "against parallel HDF5 with HDF5_MPI=ON."
                )

            self.compute_stream.synchronize()
            self.comm_stream.synchronize()
            self.mpi_comm.Barrier()

            local_host = cp.asnumpy(
                self.u[1 : self.local_ny + 1, :]
            )

            with h5py.File(
                filename, "w", driver="mpio", comm=self.mpi_comm
            ) as f:
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
                dset.attrs["backend"] = "CuPy + NCCL + mpi4py"
                dset.attrs["decomposition"] = "1D row decomposition"
                dset.attrs["communication"] = (
                    "CuPy NCCL send_recv with communication/computation overlap"
                )

                if niters is not None:
                    dset.attrs["niters"] = niters
                if rel_update_norm is not None:
                    dset.attrs["rel_update_norm"] = rel_update_norm

            self.mpi_comm.Barrier()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Distributed 2D Laplace solver using CuPy, overlapped NCCL "
            "halo exchange, mpi4py, and parallel HDF5."
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
    parser.add_argument(
        "--fixed-iters",
        type=int,
        default=None,
        help="Run exactly this many iterations, ignoring tolerance.",
    )
    return parser.parse_args()


def validate_common_args(args, mpi_comm):
    rank = mpi_comm.Get_rank()
    size = mpi_comm.Get_size()

    def abort_if(condition, message):
        if condition:
            if rank == 0:
                print(f"Error: {message}", flush=True)
            mpi_comm.Abort(1)

    abort_if(
        args.nx < 3 or args.ny < 3,
        "nx and ny must both be at least 3",
    )
    abort_if(
        not np.isfinite(args.lx) or not np.isfinite(args.ly),
        "lx and ly must be finite",
    )
    abort_if(
        args.lx <= 0.0 or args.ly <= 0.0,
        "lx and ly must be positive",
    )

    # Keep the original conservative rank limit. A remainder decomposition
    # can still produce a boundary-only end rank for some (ny, size) pairs;
    # the solver handles that case by participating in NCCL and contributing
    # zero to the local convergence reduction.
    abort_if(
        size > args.ny - 2,
        "number of MPI ranks must be <= ny - 2",
    )

    abort_if(args.max_iters < 1, "max_iters must be at least 1")
    abort_if(args.check_every < 1, "check_every must be at least 1")
    abort_if(
        args.fixed_iters is not None and args.fixed_iters < 1,
        "fixed_iters must be at least 1",
    )

    if args.fixed_iters is None:
        abort_if(
            not np.isfinite(args.tol) or args.tol <= 0.0,
            "tol must be finite and positive unless fixed_iters is used",
        )

    for name in ("bc_left", "bc_right", "bc_bottom", "bc_top"):
        abort_if(
            not np.isfinite(getattr(args, name)),
            f"{name} must be finite",
        )


def main():
    args = parse_args()
    mpi_comm = MPI.COMM_WORLD
    validate_common_args(args, mpi_comm)

    rank = mpi_comm.Get_rank()
    size = mpi_comm.Get_size()

    # One MPI rank owns one GPU.
    local_rank = get_local_rank(mpi_comm)
    ndev = cp.cuda.runtime.getDeviceCount()

    if ndev <= 0:
        if rank == 0:
            print("Error: no CUDA devices found", flush=True)
        mpi_comm.Abort(1)

    if local_rank >= ndev:
        if rank == 0:
            print(
                "Error: more local MPI ranks than visible CUDA devices.\n"
                "Use one MPI rank per GPU or explicitly implement GPU "
                "oversubscription.",
                flush=True,
            )
        mpi_comm.Abort(1)

    cp.cuda.Device(local_rank).use()

    nccl_comm = init_process_group(
        size,
        rank,
        backend="nccl",
        use_mpi=True,
    )

    if not hasattr(nccl_comm, "send_recv"):
        if rank == 0:
            print(
                "Error: this CuPy NCCL backend does not provide send_recv().",
                flush=True,
            )
            print(f"CuPy version: {cp.__version__}", flush=True)
            print(
                f"NCCL communicator type: {type(nccl_comm)}",
                flush=True,
            )
            print(
                "Use explicit grouped NCCL send/recv or adapt "
                "exchange_halos_nccl().",
                flush=True,
            )
        mpi_comm.Abort(1)

    solver = Laplace2DDistributedNCCL(
        nx=args.nx,
        ny=args.ny,
        lx=args.lx,
        ly=args.ly,
        nccl_comm=nccl_comm,
        mpi_comm=mpi_comm,
        bc_left=args.bc_left,
        bc_right=args.bc_right,
        bc_bottom=args.bc_bottom,
        bc_top=args.bc_top,
        dtype=cp.float64,
    )

    if rank == 0:
        print("Distributed CuPy/NCCL Laplace solver with overlap")
        print(f"  global grid       : ny={args.ny}, nx={args.nx}")
        print(f"  physical size     : ly={args.ly}, lx={args.lx}")
        print(f"  MPI/NCCL ranks    : {size}")
        print("  compute stream    : non-blocking")
        print("  communication     : non-blocking NCCL stream")
        print(f"  check every       : {args.check_every}")
        print(f"  output            : {args.output}")

        if args.fixed_iters is not None:
            print("  mode              : fixed iteration")
            print(f"  fixed iterations  : {args.fixed_iters}")
        else:
            print("  mode              : tolerance")
            print(f"  tolerance         : {args.tol}")
            print(f"  max iterations    : {args.max_iters}")

        print("", flush=True)

    device_id = cp.cuda.Device().id
    props = cp.cuda.runtime.getDeviceProperties(device_id)
    raw_name = props["name"]
    gpu_name = (
        raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
    )

    itemsize = cp.dtype(cp.float64).itemsize
    local_bytes_per_array = (
        (solver.local_ny + 2) * solver.nx * itemsize
    )
    global_bytes_per_array = args.ny * args.nx * itemsize

    ordered_print(
        mpi_comm,
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
        print(
            f"  global field size : "
            f"{global_bytes_per_array / 1024**2:.2f} MiB"
        )
        print("", flush=True)
        print("Warming up CuPy + NCCL overlap path...", flush=True)

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
        solver.write_parallel_hdf5(args.output, niters=niters, rel_update_norm=rel_update_norm)
    
    global_updates_per_iter = (args.nx - 2) * (args.ny - 2)
    gupdates_per_second = (global_updates_per_iter * niters / elapsed / 1.0e9 )

    if rank == 0:
        print("", flush=True)
        print(f"RawCuPy+NCCL iterations                     : {niters}")
        print(f"RawCuPy+NCCL final relative update norm     : {rel_update_norm:.6e}")
        print(f"RawCuPy+NCCL elapsed time [s]               : {elapsed:.6f}")
        print(f"RawCuPy+NCCL time per iter [ms]             : {1000.0 * elapsed / max(niters, 1):.6f}")
        print(f"RawCuPy+NCCL Jacobi throughput [GUpdates/s] : {gupdates_per_second:.6f}")
        print(f"RawCuPy+NCCL wrote solution                 : {args.output}")
        print("", flush=True)

if __name__ == "__main__":
    main()
