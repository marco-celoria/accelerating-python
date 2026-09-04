import numpy as np
import cupy as cp
import cupyx

n_elements = 1_000_000_000
n_streams  = 4
chunk_size = (n_elements + n_streams - 1) // n_streams

# Pinned host memory enables asynchronous host-device transfers.
h_a = cupyx.empty_pinned(n_elements, dtype=np.float64)
h_b = cupyx.empty_pinned(n_elements, dtype=np.float64)
h_o = cupyx.empty_pinned(n_elements, dtype=np.float64)

rng = np.random.default_rng(12345)
h_a[:] = rng.standard_normal(n_elements)
h_b[:] = rng.standard_normal(n_elements)

# Multiple non-blocking streams can permit overlap between
# host-device transfers and kernel execution.
streams = [cp.cuda.Stream(non_blocking=True) for _ in range(n_streams)]
device_buffers = []

for stream_id, stream in enumerate(streams):
    start = stream_id * chunk_size
    stop = min(start + chunk_size, n_elements)
    if start >= stop:
        continue
        
    with stream:
        d_a = cp.asarray(h_a[start:stop])
        d_b = cp.asarray(h_b[start:stop])
        d_o = 0.5 * cp.hypot(d_a, d_b)
        d_o.get(out=h_o[start:stop], blocking=False,)
        # Retain explicit references to the device arrays until all
        # asynchronous operations that use them have completed.
        device_buffers.append((d_a, d_b, d_o))

for stream in streams:
    stream.synchronize()

expected = 0.5 * np.hypot(h_a, h_b)
np.testing.assert_allclose(h_o, expected, rtol=1.0e-12, atol=1.0e-12)
print("Verification passed.")
