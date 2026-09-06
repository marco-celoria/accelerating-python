import math
import numpy as np
from numba import cuda

@cuda.jit
def scaled_hypot_kernel(a, b, out):
    i = cuda.grid(1)
    if i < out.size:
        out[i] = 0.5 * math.hypot(a[i], b[i])

n_elements = 1_000_000_000
n_streams = 4
chunk_size = (n_elements + n_streams - 1) // n_streams

threads_per_block = 256

# Pinned host memory enables asynchronous host-device transfers.
h_a = cuda.pinned_array(n_elements, dtype=np.float64)
h_b = cuda.pinned_array(n_elements, dtype=np.float64)
h_o = cuda.pinned_array(n_elements, dtype=np.float64)

rng = np.random.default_rng(12345)
h_a[:] = rng.standard_normal(n_elements)
h_b[:] = rng.standard_normal(n_elements)

# Independent CUDA streams may allow overlap between host-device
# transfers, kernel execution, and device-host transfers.
streams = [cuda.stream() for _ in range(n_streams)]

# Keep explicit references to device allocations until all asynchronous
# operations using them have completed.
device_buffers = []

for stream_id, stream in enumerate(streams):
    start = stream_id * chunk_size
    stop = min(start + chunk_size, n_elements)
    if start >= stop:
        continue

    n_chunk = stop - start
    blocks_per_grid = (n_chunk + threads_per_block - 1) // threads_per_block

    # Enqueue asynchronous host-to-device transfers.
    d_a = cuda.to_device(h_a[start:stop], stream=stream)
    d_b = cuda.to_device(h_b[start:stop], stream=stream)

    # Allocate the output in device memory.
    d_o = cuda.device_array(n_chunk, dtype=np.float64, stream=stream)

    # The kernel launch is asynchronous and ordered after the H2D
    # transfers because it is submitted to the same stream.
    scaled_hypot_kernel[blocks_per_grid, threads_per_block, stream](d_a, d_b, d_o)

    # Enqueue an asynchronous device-to-host transfer into the
    # appropriate slice of the pinned output array.
    d_o.copy_to_host(ary=h_o[start:stop], stream=stream)
    device_buffers.append((d_a, d_b, d_o))

# Wait until all operations in every stream have completed.
for stream in streams:
    stream.synchronize()

expected = 0.5 * np.hypot(h_a, h_b)
np.testing.assert_allclose(h_o, expected, rtol=1.0e-12, atol=1.0e-12)
print("Verification passed.")
