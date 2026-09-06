import numpy as np
from numba import cuda

@cuda.jit
def add_2d(a, b, c):
    col, row = cuda.grid(2)
    if row < a.shape[0] and col < a.shape[1]:
        c[row, col] = a[row, col] + b[row, col]
        

# Initialize the input arrays in host memory.
rows = 1000
cols = 2000
h_a = np.full((rows, cols), 1.5, dtype=np.float32)
h_b = np.full((rows, cols), 2.5, dtype=np.float32)
# Allocate device memory and transfer the input arrays.
d_a = cuda.to_device(h_a)
d_b = cuda.to_device(h_b)
d_c = cuda.device_array_like(d_a)

# Configure a two-dimensional thread block.
threads_per_block = (16, 16)
# Round up the grid dimensions to cover the complete array.
blocks_per_grid = (
    (cols + threads_per_block[0] - 1) // threads_per_block[0],
    (rows + threads_per_block[1] - 1) // threads_per_block[1],
)

# Warm up the kernel to trigger compilation and initialization.
add_2d[blocks_per_grid, threads_per_block](d_a, d_b, d_c)
cuda.synchronize()

# Measure the cached kernel using CUDA events.
start_event = cuda.event()
stop_event  = cuda.event()
start_event.record()

add_2d[blocks_per_grid, threads_per_block](d_a, d_b, d_c)

stop_event.record()
stop_event.synchronize()
elapsed_ms = cuda.event_elapsed_time(start_event, stop_event)

# Copy the result back to host memory.
h_c = d_c.copy_to_host()

# Improved assertion with 2D coordinate diagnostics
np.testing.assert_allclose(h_c, h_a + h_b, rtol=1.0e-6, atol=1.0e-6)
print(f"Kernel execution time: {elapsed_ms:.3f} ms")
print("Verification passed.")
