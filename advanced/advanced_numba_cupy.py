import numpy as np
import cupy as cp
from numba import cuda

# Pinned host memory enables asynchronous D->H transfers.
h_x = cuda.pinned_array(5, dtype=np.float32)
h_x[:] = np.arange(5, dtype=np.float32)
res_cupy = cuda.pinned_array_like(h_x)
res_numba = cuda.pinned_array_like(h_x)

# Create a CUDA stream in Numba and expose it to CuPy.
numba_stream = cuda.stream()
cupy_stream = cp.cuda.Stream.from_external(numba_stream)

# Transfer input to the GPU on the shared stream.
d_x_numba = cuda.to_device(h_x, stream=numba_stream,)

# Create a zero-copy CuPy view of the same device allocation.
d_x_cupy = cp.asarray(d_x_numba)

# Schedule CuPy work on the same CUDA stream.
with cupy_stream:
    d_x_cupy *= cp.float32(10.0)
    # Enqueue an asynchronous copy into pinned host memory.
    d_x_cupy.get(out=res_cupy, stream=cupy_stream, blocking=False)

# Queue a second D->H transfer after the CuPy operation.
# Same CUDA stream enforces dependencies 
# without manual synchronization.
d_x_numba.copy_to_host(ary=res_numba, stream=numba_stream,)

# Synchronize only before the host consumes the results.
numba_stream.synchronize()
ref_numpy = h_x * np.float32(10.0)
np.testing.assert_array_equal(res_cupy, ref_numpy)
np.testing.assert_array_equal(res_numba, ref_numpy)
print("Verification passed.")
