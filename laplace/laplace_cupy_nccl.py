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


# NCCL is used for GPU-GPU communication and collectives.
# This implementation requires a CuPy NCCLBackend version that provides
# the non-standard or version-specific send_recv() convenience method.
#
# MPI is used for process launch, rank discovery, timing aggregation, and
# parallel HDF5.
#
# The RawKernels below assume contiguous C-order arrays. Do not pass
# transposed or otherwise strided views unless the kernels are updated to
# account for strides.

TPB_X = 32
TPB_Y = 8

REDUCE_TPB = 256
REDUCE_BLOCKS_PER_SM = 4

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
    const double down  = u[idx - nx];
    const double up    = u[idx + nx];

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

        u[0, :]                 lower halo
        u[1:local_ny + 1, :]    owned physical rows
        u[local_ny + 1, :]      upper halo

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

        self.down = self.rank - 1 if self.rank > 0 else None
        self.up   = self.rank + 1 if self.rank < self.size - 1 else None
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

        self.update_i0, self.update_i1 = (self._owned_update_row_bounds())
        self.local_update_count = (max(0, self.update_i1 - self.update_i0) * (self.nx - 2))
        self.jacobi_threads = (TPB_X, TPB_Y)
        self.rel_update_norm_threads = REDUCE_TPB
        if (self.rel_update_norm_threads < 1 or (self.rel_update_norm_threads & (self.rel_update_norm_threads - 1)) != 0):
            raise ValueError("rel_update_norm_threads must be a positive power of two")
        if REDUCE_BLOCKS_PER_SM < 1:
            raise ValueError("REDUCE_BLOCKS_PER_SM must be positive")
        attributes = cp.cuda.Device().attributes
        self.num_sms = attributes["MultiProcessorCount"]
        if self.local_update_count > 0:
            required_blocks = (self.local_update_count + self.rel_update_norm_threads - 1) // self.rel_update_norm_threads
            self.rel_update_norm_blocks = min(required_blocks, REDUCE_BLOCKS_PER_SM * self.num_sms)
        else:
            self.rel_update_norm_blocks = 0
        self.rel_update_norm_workspace_size = max(1, self.rel_update_norm_blocks,)
        self.rel_update_norm_shared_bytes = (2 * self.rel_update_norm_threads * cp.dtype(cp.float64).itemsize)

        self.nx_kernel = np.uint64(self.nx)
        self.update_i0_kernel = np.uint64(self.update_i0)
        self.update_i1_kernel = np.uint64(self.update_i1)
        self.local_update_count_kernel = np.uint64(self.local_update_count)
        self.dx2_kernel = np.float64(self.dx2)
        self.dy2_kernel = np.float64(self.dy2)
        self.inv_denom_kernel = np.float64(self.inv_denom)
        self.lower_interface = (self.update_i0 if self.down is not None else None)
        self.upper_interface = (self.update_i1 - 1 if self.up is not None else None)
        self.deep_i0 = (self.update_i0 + int(self.down is not None))
        self.deep_i1 = (self.update_i1 - int(self.up is not None))
        self._validate_kernel_configuration()

        # Allocate and initialize on the compute stream. The recorded event
        # makes this initialization visible to the communication stream.
        with self.compute_stream:
            self.u = cp.zeros((self.local_ny + 2, self.nx), dtype=self.dtype,)
            self.unew = cp.zeros_like(self.u)
            self._apply_dirichlet_bc(self.u)
            self._apply_dirichlet_bc(self.unew)
            self.red_local  = cp.zeros(2, dtype=cp.float64,)
            self.red_global = cp.zeros(2, dtype=cp.float64,)
            self.partial_diff = cp.empty(self.rel_update_norm_workspace_size, dtype=cp.float64,)
            self.partial_norm = cp.empty(self.rel_update_norm_workspace_size, dtype=cp.float64,)
            self.u_ready_event.record(self.compute_stream)


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
            raise ValueError("This RawKernel implementation currently supports only cp.float64")

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
        if (fixed_iters is not None and not isinstance(fixed_iters, integer_types,)):
            raise TypeError("fixed_iters must be an integer")
        if max_iters < 1:
            raise ValueError("max_iters must be at least 1")
        if check_every < 1:
            raise ValueError("check_every must be at least 1")
        if (fixed_iters is not None and fixed_iters < 1):
            raise ValueError("fixed_iters must be at least 1")
        if fixed_iters is None:
            if (not np.isfinite(float(tol)) or float(tol) <= 0.0):
                raise ValueError("tol must be finite and positive unless fixed_iters is used")

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
            if self.down is not None:
                self.nccl_comm.send_recv(
                    in_array=self.u[1, :],
                    out_array=self.u[0, :],
                    peer=self.down,
                    stream=stream,
                )

            if self.up is not None:
                self.nccl_comm.send_recv(
                    in_array=self.u[self.local_ny, :],
                    out_array=self.u[self.local_ny + 1, :],
                    peer=self.up,
                    stream=stream,
                )

    def _launch_jacobi_rows(self, i0, i1, stream):
        """Enqueue Jacobi updates for local rows ``[i0, i1)``."""
        if i0 >= i1:
            return
        
        blocks = (
            (self.nx - 2 + self.jacobi_threads[0] - 1) // self.jacobi_threads[0],
            (i1 - i0 + self.jacobi_threads[1] - 1) // self.jacobi_threads[1],
        )

        jacobi_kernel(
            blocks,
            self.jacobi_threads,
            (
                self.u,
                self.unew,
                self.nx_kernel,
                np.uint64(i0),
                np.uint64(i1),
                self.dx2_kernel,
                self.dy2_kernel,
                self.inv_denom_kernel,
            ),
            stream=stream,
        )

    def _enqueue_overlapped_jacobi_step(self):
        """
        Enqueue one Jacobi step with communication/computation overlap.

        The deep interior does not depend on received halo rows. 
        The first updated row needs the lower halo when a down rank exists,
        and the last updated row needs the upper halo when an up rank exists.
        """
        with nvtx_range("jacobi_step_overlapped"):
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
            if self.update_i0 >= self.update_i1:
                # Preserve NCCL ordering before a possible all-reduce and
                # before publishing the next u_ready_event.
                self.compute_stream.wait_event(self.halos_ready_event)
                return
            with nvtx_range("jacobi_deep_interior"):
                self._launch_jacobi_rows(self.deep_i0, self.deep_i1, self.compute_stream,)

            # Only halo-dependent kernels wait. Previously enqueued deep
            # interior work remains eligible to overlap with NCCL.
            self.compute_stream.wait_event(self.halos_ready_event)

            with nvtx_range("jacobi_interfaces"):
                if self.upper_interface is not None:
                    self._launch_jacobi_rows(self.upper_interface, self.upper_interface + 1, self.compute_stream,)
                if (self.lower_interface is not None and self.lower_interface != self.upper_interface):
                    self._launch_jacobi_rows(self.lower_interface, self.lower_interface + 1, self.compute_stream,)

    def compute_local_reduction_rawkernel(self, stream):
        """
        Enqueue the local partial norm reductions on stream.

        This method requires local_update_count > 0.
        """
        rel_update_norm_partial_kernel(
            (self.rel_update_norm_blocks,), 
            (self.rel_update_norm_threads,),
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
            stream=stream,
        )
        with stream:
            local_diff2 = cp.sum(self.partial_diff[ : self.rel_update_norm_blocks])
            local_norm2 = cp.sum(self.partial_norm[ : self.rel_update_norm_blocks])
        return local_diff2, local_norm2


    def compute_relative_l2_update_norm(self):
        """
        Compute the global relative L2 iterate-update norm:
            ||u_new - u||_2 / ||u_new||_2
        over the global physical interior.
        This convergence indicator is not the discrete Laplace residual.
        This method is called after all Jacobi kernels have been enqueued on compute_stream. 
        Stream ordering therefore makes the reduction wait
        for the complete update without a device-wide synchronization.
        """
        with nvtx_range("compute_relative_l2_update_norm"):
            stream = self.compute_stream
            if self.local_update_count > 0:
                local_diff2, local_norm2 = (self.compute_local_reduction_rawkernel(stream))

                with stream:
                    self.red_local[0] = local_diff2
                    self.red_local[1] = local_norm2
            else:
                with stream:
                    self.red_local.fill(0.0)

            self.nccl_comm.all_reduce(
                self.red_local,
                self.red_global,
                op="sum",
                stream=stream,
            )

            stream.synchronize()
            diff2, norm2 = cp.asnumpy(self.red_global)
            return float(np.sqrt(diff2) / max(np.sqrt(norm2), 1.0e-300))

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
        iteration_limit = fixed_iters if fixed_iters is not None else max_iters

        with nvtx_range("run"):

            start_event = cp.cuda.Event()
            stop_event = cp.cuda.Event()
            self.compute_stream.synchronize()
            self.comm_stream.synchronize()
            self.mpi_comm.Barrier()
            wall_t0 = MPI.Wtime()

            # The initial field is already represented by u_ready_event.
            # Start timing on compute_stream and make comm_stream wait on the
            # same timing origin.
            start_event.record(self.compute_stream)
            self.comm_stream.wait_event(start_event)

            for it in range(1, iteration_limit + 1):
                self._enqueue_overlapped_jacobi_step()

                do_check = (it == 1 or it % check_every == 0 or it == iteration_limit)
                converged = False
                if do_check:
                    rel_update_norm = (self.compute_relative_l2_update_norm())
                    converged = (fixed_iters is None and rel_update_norm < tol)
                # Always finish and swap exactly once per iteration. Thus
                # self.u always contains the latest completed/enqueued field.
                self._finish_iteration()
                niters = it
                if converged:
                    break

            # Every communication operation needed by the final update is an
            # ancestor of compute_stream through halos_ready_event.
            stop_event.record(self.compute_stream)
            stop_event.synchronize()
            elapsed_cuda_local = (cp.cuda.get_elapsed_time(start_event, stop_event) * 1.0e-3)
            elapsed_wall_local = MPI.Wtime() - wall_t0
            elapsed_cuda_max = self.mpi_comm.allreduce(elapsed_cuda_local, op=MPI.MAX,)
            elapsed_wall_max = self.mpi_comm.allreduce(elapsed_wall_local, op=MPI.MAX,)
            return niters, rel_update_norm, elapsed_cuda_max, elapsed_wall_max


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
            # Ensure parent directory exists across rank 0 (or all ranks safely)
            if self.rank == 0:
                out_dir = os.path.dirname(filename)
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
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
                dset.attrs["communication"] = ("CuPy NCCL send_recv with communication/computation overlap")
                dset.attrs["global_reduction"] = ("NCCL AllReduce on two device FP64 values")
                dset.attrs["overlap"] = ("deep-interior Jacobi overlaps NCCL halo exchange")
                dset.attrs["convergence_metric"] = ("relative L2 iterate-update norm over global physical interior")
                dset.attrs["rel_update_norm_method"] = ("CUDA partial reduction + CuPy reduction + NCCL AllReduce")
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
    parser.add_argument("--output", type=str, default=None, help="Optional parallel HDF5 output filename.",)
    parser.add_argument("--fixed-iters", type=int, default=None,
        help=("Run exactly this many iterations without tolerance-based termination. "
              "The update norm is still evaluated according to --check-every."),
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

    abort_if(args.nx < 3 or args.ny < 3, "nx and ny must both be at least 3",)
    abort_if(not np.isfinite(args.lx) or not np.isfinite(args.ly), "lx and ly must be finite",)
    abort_if(args.lx <= 0.0 or args.ly <= 0.0, "lx and ly must be positive",)
    # Full-row decomposition permits at most ny ranks.
    # Boundary-only ranks are supported but perform no Jacobi updates.
    abort_if(size > args.ny, "number of MPI ranks must be <= ny for full-row decomposition",)
    abort_if(args.max_iters < 1, "max_iters must be at least 1")
    abort_if(args.check_every < 1, "check_every must be at least 1")
    abort_if(args.fixed_iters is not None and args.fixed_iters < 1, "fixed_iters must be at least 1",)

    if args.fixed_iters is None:
        abort_if(not np.isfinite(args.tol) or args.tol <= 0.0, "tol must be finite and positive unless fixed_iters is used",)

    for name in ("bc_left", "bc_right", "bc_bottom", "bc_top"):
        abort_if(not np.isfinite(getattr(args, name)), f"{name} must be finite",)


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

    # One MPI rank owns one GPU.
    try:
        local_rank = get_local_rank(mpi_comm)
        ndev = cp.cuda.runtime.getDeviceCount()
    except Exception as exc:
        print(f"rank {rank}: CUDA/local-rank discovery failed: {exc}", flush=True,)
        mpi_comm.Abort(1)

    if ndev <= 0:
        if rank == 0:
            print("Error: no CUDA devices found", flush=True)
        mpi_comm.Abort(1)
    
    device_id = 0 if ndev == 1 else local_rank
    if device_id >= ndev:
        print(f"rank {rank}: cannot map local_rank={local_rank} to {ndev} visible GPU(s)", flush=True,)
        mpi_comm.Abort(1)

    try:
        cp.cuda.Device(device_id).use()
        nccl_comm = init_process_group(
            size,
            rank,
            backend="nccl",
            use_mpi=True,
        )
    except Exception as exc:
        print(f"rank {rank}: NCCL initialization failed: {exc}", flush=True,)
        mpi_comm.Abort(1)
    
    if rank == 0:
        print(f"CuPy version: {cp.__version__}", flush=True)
        print(f"NCCL communicator type: {type(nccl_comm).__name__}",flush=True,)

    if not hasattr(nccl_comm, "send_recv"):
        if rank == 0:
            print("Error: this CuPy NCCL backend does not provide send_recv().", flush=True,)
            print("Use explicit grouped NCCL send/recv or adapt exchange_halos_nccl().", flush=True,)
        mpi_comm.Abort(1)
    
    try:
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
    except Exception as exc:
        print(f"rank {rank}: solver initialization failed: {exc}", flush=True,)
        mpi_comm.Abort(1)

    local_deep_rows = max(0, solver.deep_i1 - solver.deep_i0,)
    min_deep_rows = mpi_comm.allreduce(local_deep_rows, op=MPI.MIN,)
    has_zero_update_rank = mpi_comm.allreduce(int(solver.local_update_count == 0), op=MPI.MAX,)
    if rank == 0 and has_zero_update_rank:
        print("Warning: one or more ranks own no global interior rows and perform no Jacobi updates.", flush=True,)
    elif rank == 0 and min_deep_rows == 0:
        print("Warning: at least one rank has no deep-interior rows; communication/computation overlap will be limited.", flush=True,)
    
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

    try:
        device_id = cp.cuda.Device().id
        props = cp.cuda.runtime.getDeviceProperties(device_id)
        raw_name = props["name"]
        gpu_name = (raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name))
    except Exception as exc:
        print(f"rank {rank}: failed to query CUDA device properties: {exc}", flush=True,)
        mpi_comm.Abort(1)

    itemsize = np.dtype(np.float64).itemsize
    local_field_bytes = ((solver.local_ny + 2) * solver.nx * itemsize)
    partial_bytes = (2 * solver.rel_update_norm_workspace_size * itemsize)
    reduction_buffer_bytes = 4 * itemsize
    explicit_device_bytes = (2 * local_field_bytes + partial_bytes + reduction_buffer_bytes)
    global_bytes_per_array = args.ny * args.nx * itemsize

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
            f"local array={solver.u.shape} | "
            f"local_updates={solver.local_update_count} | "
            f"update_rows={solver.update_i1 - solver.update_i0} | "
            f"reduce_tpb={solver.rel_update_norm_threads} | "
            f"reduce_blocks={solver.rel_update_norm_blocks} | "
            f"reduce_blocks_per_sm={REDUCE_BLOCKS_PER_SM} | "
            f"norm_workspace={partial_bytes / 1024:.2f} KiB | "
            f"memory/field={local_field_bytes / 1024**2:.2f} MiB | "
            f"explicit_device={explicit_device_bytes / 1024**2:.2f} MiB"
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
    
    try:
        solver.warmup()
    except Exception as exc:
        print(f"rank {rank}: NCCL warm-up failed: {exc}", flush=True,)
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
    total_updates = global_updates_per_iter * niters
    gupdates_per_second_cuda_event = (total_updates / elapsed_cuda_event / 1.0e9 if elapsed_cuda_event > 0.0 else float("inf"))
    gupdates_per_second_wall       = (total_updates / elapsed_wall       / 1.0e9 if elapsed_wall       > 0.0 else float("inf"))

    if rank == 0:
        print()
        print("============================================================")
        print("Solver summary")
        print("============================================================")
        print("Implementation                                                : CuPy RawKernel Jacobi + overlapped NCCL communication")
        print(f"CuPy RawKernel + NCCL iterations                              : {niters}")
        if args.fixed_iters is None:
            print(f"CuPy RawKernel + NCCL converged                               : {converged}")
        else:
            print("CuPy RawKernel + NCCL convergence used to stop                : no")
        print(f"CuPy RawKernel + NCCL final relative iterate-update norm      : {rel_update_norm:.6e}")
        print(f"CuPy RawKernel + NCCL CUDA-event interval [s]                 : {elapsed_cuda_event:.6f}")
        print(f"CuPy RawKernel + NCCL wall elapsed time [s]                   : {elapsed_wall:.6f}")
        print(f"CuPy RawKernel + NCCL CUDA-event time/iter [ms]               : {1000.0 * elapsed_cuda_event / max(niters, 1):.6f}")
        print(f"CuPy RawKernel + NCCL wall time/iter [ms]                     : {1000.0 * elapsed_wall / max(niters, 1):.6f}")
        print("Throughput metric: effective monitored distributed solver")
        print(f"CuPy RawKernel + NCCL CUDA-event throughput [GCell-updates/s] : {gupdates_per_second_cuda_event:.6f}")
        print(f"CuPy RawKernel + NCCL wall throughput [GCell-updates/s]       : {gupdates_per_second_wall:.6f}")
    if args.output is not None:
        try:
            solver.write_parallel_hdf5(args.output, niters=niters, rel_update_norm=rel_update_norm,)
        except Exception as exc:
            print(f"rank {rank}: parallel HDF5 output failed: {exc}", flush=True,)
            mpi_comm.Abort(1)
        if rank == 0:
            print(f"CuPy RawKernel + NCCL wrote solution                          : {args.output}", flush=True,)
    if rank == 0:
        print("", flush=True)
    try:
        solver.compute_stream.synchronize()
        solver.comm_stream.synchronize()
        mpi_comm.Barrier()
        if hasattr(nccl_comm, "destroy"):
            nccl_comm.destroy()
        elif hasattr(nccl_comm, "close"):
            nccl_comm.close()
    except Exception as exc:
        print(f"rank {rank}: NCCL cleanup failed: {exc}", flush=True,)
        mpi_comm.Abort(1)

if __name__ == "__main__":
    main()


