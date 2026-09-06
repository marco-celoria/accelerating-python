import cupy as cp
from numba import cuda

@cuda.jit
def add_numba_kernel(x, y, result):
    i = cuda.grid(1)
    if i < result.size:
        result[i] = x[i] + y[i]

# Create and own the CUDA stream with CuPy.
# Work launched on cupy_stream or numba_stream executes asynchronously 
# relative to the default stream.
cupy_stream = cp.cuda.Stream(non_blocking=True)

# Wrap the same CUDA stream for use by Numba.
# CuPy remains responsible for the stream's lifetime.
numba_stream = cuda.external_stream(cupy_stream.ptr)

# Allocate and initialize the arrays on the shared stream.
with cupy_stream:
    d_a = cp.arange(10, dtype=cp.float64)
    d_b = 2.0 * d_a
    d_o = cp.empty_like(d_a)

# Configure a one-dimensional launch.
threads = 128
blocks = (d_o.size + threads - 1) // threads

# Launch the Numba kernel on the shared stream.
add_numba_kernel[blocks, threads, numba_stream](d_a, d_b, d_o)

# Submit dependent CuPy operations to the same CUDA stream.
with cupy_stream:
    d_res = cp.sqrt(d_o).sum()
    d_ref = cp.sqrt(d_a + d_b).sum()

# Wait only when the host needs the results.
cupy_stream.synchronize()
cp.testing.assert_allclose(d_res, d_ref, rtol=1.0e-12, atol=1.0e-12)
print("Verification passed.")
