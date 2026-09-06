import numpy as np
from numba import cuda

@cuda.jit
def add(a, b, c):
    i = cuda.grid(1)
    if i < c.size:
        c[i] = a[i] + b[i]

# Initialize the input arrays in host memory.
n_elements = 10_000_000
h_x = np.arange(n_elements, dtype=np.float32)
h_y = np.random.random(n_elements).astype(np.float32)

# Allocate device memory and transfer the input arrays.
d_x = cuda.to_device(h_x)
d_y = cuda.to_device(h_y)
d_z = cuda.device_array_like(d_x)

# Configure a one-dimensional execution grid.
threads_per_block = 256
blocks_per_grid = (n_elements + threads_per_block - 1) // threads_per_block
# 39_063 blocks in the grid each with 256 threads 
# 10_000_128 total threads in the grid
add[blocks_per_grid, threads_per_block](d_x, d_y, d_z)

# Copying the result to the host waits for the kernel to complete.
h_z = d_z.copy_to_host()

# Verify the result on the host.
np.testing.assert_allclose(h_z, h_x + h_y, rtol=1.0e-6, atol=1.0e-6)
print("Verification passed.")
