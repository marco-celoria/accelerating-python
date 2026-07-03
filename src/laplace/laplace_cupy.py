#!/usr/bin/env python3

import argparse
import cupy as cp
import numpy as np

class Laplace2DCuPy:
    """
    Serial CuPy Jacobi solver (GPU offload) matching the NumPy implementation.
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
        Laplace2DCuPy._validate_problem_parameters(nx, ny, lx, ly, bc_left, bc_right, bc_bottom, bc_top)
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
        self.u = cp.zeros((self.ny, self.nx), dtype=cp.float64)
        self.unew = cp.zeros_like(self.u)
        self._apply_dirichlet_bc(self.u)
        self._apply_dirichlet_bc(self.unew)


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
    def _apply_dirichlet_bc(self, arr):
        # same order as NumPy + distributed solver
        cp.cuda.nvtx.RangePush("_apply_dirichlet_bc")
        try:
            arr[ :,  0] = self.bc_left
            arr[ :, -1] = self.bc_right
            arr[ 0,  :] = self.bc_bottom
            arr[-1,  :] = self.bc_top
        finally:
            cp.cuda.nvtx.RangePop()

    # Compute kernel (vectorized)
    def jacobi_step(self):
        cp.cuda.nvtx.RangePush("jacobi_step")
        try:
            # Vectorized CuPy is productive and readable, 
            # but for stencil kernels it may generate multiple memory passes. 
            # A custom kernel can reduce temporaries and improve memory bandwidth use.
            self.unew[1:-1, 1:-1] = self.inv_denom * (
                      (self.u[1:-1, :-2] + self.u[1:-1, 2:]) * self.dy2
                    + (self.u[:-2, 1:-1] + self.u[2:, 1:-1]) * self.dx2)
            # Boundaries are initialized once in both u and unew.
            # Since this update only writes: self.unew[1:-1, 1:-1]
            # the boundary entries of unew remain unchanged after initialization.
            # Since jacobi_step writes only interior points, boundaries remain valid across ping-pong swaps.
            #self._apply_dirichlet_bc(self.unew) <- Not needed because only interior points are overwritten.
        finally:
            cp.cuda.nvtx.RangePop()

    # Relative update norm
    def compute_relative_l2_update_norm(self):
        cp.cuda.nvtx.RangePush("compute_relative_l2_update_norm")
        rel_update_norm = 0.0
        try:
            diff = self.unew[1:-1, 1:-1] - self.u[1:-1, 1:-1]
            interior = self.unew[1:-1, 1:-1]
            diff2 = cp.sum(diff * diff)
            norm2 = cp.sum(interior * interior)
            rel_update_norm_gpu = cp.sqrt(diff2) / cp.maximum(cp.sqrt(norm2), 1.0e-300)
            rel_update_norm = rel_update_norm_gpu.get()
        finally:
            cp.cuda.nvtx.RangePop()
        return float(rel_update_norm)

    # Solver loop
    def run(self, max_iters, tol, check_every, fixed_iters=None):
        cp.cuda.nvtx.RangePush("run")
        rel_update_norm = float("inf")
        niters = 0
        elapsed = 0.0
        try:
            Laplace2DCuPy._validate_run_parameters(max_iters, tol, check_every, fixed_iters)
            nstop = fixed_iters if fixed_iters is not None else max_iters
            start = cp.cuda.Event()
            stop = cp.cuda.Event()
            # Make sure all previous GPU work is finished before timing.
            cp.cuda.get_current_stream().synchronize()
            start.record()
            for it in range(1, nstop + 1):
                self.jacobi_step()
                do_check = (it == 1 or it % check_every == 0 or it == nstop)
                if do_check:
                    rel_update_norm = self.compute_relative_l2_update_norm()
                    # Device scalars must be copied to the host before Python control flow can use them.
                    if fixed_iters is None and rel_update_norm < tol:
                        self.u, self.unew = self.unew, self.u
                        niters = it
                        break
                self.u, self.unew = self.unew, self.u
                niters = it
            stop.record()
            stop.synchronize()
            elapsed = cp.cuda.get_elapsed_time(start, stop) / 1000.0
        finally:
            cp.cuda.nvtx.RangePop()
        return niters, rel_update_norm, elapsed

    # Output
    def write_hdf5(self, filename, niters, rel_update_norm):
        cp.cuda.nvtx.RangePush("write_hdf5")
        try:
            import h5py
            cp.cuda.get_current_stream().synchronize()
            with h5py.File(filename, "w") as f:
                dset = f.create_dataset("u", data=cp.asnumpy(self.u))
                dset.attrs["nx"] = self.nx
                dset.attrs["ny"] = self.ny
                dset.attrs["lx"] = self.lx
                dset.attrs["ly"] = self.ly
                dset.attrs["bc_left"] = self.bc_left
                dset.attrs["bc_right"] = self.bc_right
                dset.attrs["bc_bottom"] = self.bc_bottom
                dset.attrs["bc_top"] = self.bc_top
                dset.attrs["solver"] = "Jacobi"
                dset.attrs["backend"] = "CuPy"
                dset.attrs["niters"] = niters
                dset.attrs["rel_update_norm"] = float(rel_update_norm)
                dset.attrs["dx"] = self.dx
                dset.attrs["dy"] = self.dy
                dset.attrs["dtype"] = str(self.u.dtype)
        finally:
            cp.cuda.nvtx.RangePop()


# CLI
def parse_args():
    parser = argparse.ArgumentParser(
        description="Serial CuPy Laplace solver (GPU) for validating distributed code."
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
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--fixed-iters", type=int, default=None,
        help="Run exactly this many iterations, ignoring tolerance.",
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
    bytes_per_array = args.ny * args.nx * cp.dtype(cp.float64).itemsize
    print("Serial CuPy Laplace solver")
    print(f"  grid              : ny={args.ny}, nx={args.nx}")
    print(f"  physical size     : ly={args.ly}, lx={args.lx}")
    print(f"  check every       : {args.check_every}")
    print(f"  GPU device name   : {name}")
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
    solver = Laplace2DCuPy(
        nx=args.nx,
        ny=args.ny,
        lx=args.lx,
        ly=args.ly,
        bc_left=args.bc_left,
        bc_right=args.bc_right,
        bc_bottom=args.bc_bottom,
        bc_top=args.bc_top,
    )
    niters, rel_update_norm, elapsed = solver.run(
        max_iters=args.max_iters,
        tol=args.tol,
        check_every=args.check_every,
        fixed_iters=args.fixed_iters,
    )
    print(f"iterations                : {niters}")
    print(f"final relative update norm: {rel_update_norm:.6e}")
    print(f"CuPy elapsed time [s]     : {elapsed:.6f}")
    print(f"CuPy time per iter [ms]   : {1000.0 * elapsed / max(niters, 1):.6f}")
    if args.output is not None:
        solver.write_hdf5(args.output, niters, rel_update_norm)
        print(f"wrote solution            : {args.output}")
    print("", flush=True)


if __name__ == "__main__":
    main()


