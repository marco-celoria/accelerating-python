import numpy as np
import cupy as cp

jacobi_kernel = cp.RawKernel(r'''
extern "C"
__global__
void jacobi_2d(
    const double* __restrict__ u,
    double* __restrict__ unew,
    const size_t n
)
{
    // Compute 2D global thread indices.
    // The +1 offset skips the first row/column, which are fixed boundaries.
    const size_t j = blockDim.x * blockIdx.x + threadIdx.x + 1;
    const size_t i = blockDim.y * blockIdx.y + threadIdx.y + 1;

    // Exclude the high-index physical boundaries and any padded
    // thread coordinates beyond the physical interior.
    if (j >= n - 1 || i >= n - 1) {
        return;
    }
    
    // Flatten the 2D coordinates (i, j) into a 1D memory offset.
    const size_t idx = i * n + j;

    // Gather the four neighboring values (left, right, down, up).
    const double left  = u[idx - 1];
    const double right = u[idx + 1];
    const double down  = u[idx - n];
    const double up    = u[idx + n];

    // Compute the stencil average and write to the output array.
    unew[idx] = 0.25 * (left + right + down + up);
}
''', "jacobi_2d", )

n = 10_000
u = cp.random.random((n, n), dtype=cp.float64)
left  = u[1:-1, :-2]
right = u[1:-1, 2:]
down  = u[:-2, 1:-1]
up    = u[2:, 1:-1]

# Both output fields are initialized to zero, including their boundaries.
raw_res  = cp.zeros_like(u)
cupy_res = cp.zeros_like(u)
threads = (32, 8)

# Smallest two-dimensional block grid that covers the physical interior.
blocks  = (
    (n - 2 + threads[0] - 1) // threads[0],
    (n - 2 + threads[1] - 1) // threads[1],
)

# Warm up the RawKernel to trigger compilation and caching.
cupy_res[1:-1, 1:-1] = 0.25 * (left + right + down + up)
jacobi_kernel(blocks, threads, (u, raw_res, np.uint64(n),))
cp.cuda.get_current_stream().synchronize()
cupy_res.fill(0.0)
raw_res.fill(0.0)

# Profile the high-level CuPy array expression.
cp.cuda.nvtx.RangePush("CuPy array-expression Jacobi iteration")
cupy_res[1:-1, 1:-1] = 0.25 * (left + right + down + up)
cp.cuda.get_current_stream().synchronize()
cp.cuda.nvtx.RangePop()

# Profile the cached custom kernel.
cp.cuda.nvtx.RangePush("CuPy RawKernel Jacobi iteration")
jacobi_kernel(blocks, threads, (u, raw_res, np.uint64(n),))
cp.cuda.get_current_stream().synchronize()
cp.cuda.nvtx.RangePop()

# Verify the result outside the profiled regions.
cp.testing.assert_allclose(raw_res, cupy_res, rtol=1.0e-12, atol=1.0e-12)
print("Verification passed.")
