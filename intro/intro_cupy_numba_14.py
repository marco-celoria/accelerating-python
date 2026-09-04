import numpy as np
from numba import cuda

@cuda.jit
def add_grid_stride(d_a, d_b, d_c):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(start, d_c.size, stride):
        d_c[i] = d_a[i] + d_b[i]

# Initialize host arrays.
n_elements = 10_000_000
h_x = np.arange(n_elements, dtype=np.float32)
h_y = np.full(n_elements, 2.0, dtype=np.float32)

# Allocate device arrays and transfer the inputs.
d_x = cuda.to_device(h_x)
d_y = cuda.to_device(h_y)
d_z = cuda.device_array_like(d_x)

threads_per_block = 256
blocks_per_grid = 128

# Launch the kernel on the GPU with 256*128=32_768 threads in the grid 
# for a much larger problem size (32_768 < 10_000_000)
add_grid_stride[blocks_per_grid, threads_per_block](d_x, d_y, d_z)

# Wait for completion and copy the result to the host.
h_z = d_z.copy_to_host()

# Improved assertion with explicit tolerances
np.testing.assert_allclose(h_z, h_x + h_y, rtol=1.0e-6, atol=1.0e-6)
print("Verification passed.")
