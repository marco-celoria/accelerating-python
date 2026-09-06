import numpy as np
from numba import cuda

@cuda.reduce
def sum_reduce(a, b):
    return a + b

h_a = np.arange(1, 1234, dtype=np.float64)
d_a = cuda.to_device(h_a)

# Perform the reduction on the GPU.
res = sum_reduce(d_a)

# Compute a reference result on the CPU.
ref = np.sum(h_a, dtype=np.float64)

# Verify the result against CPU reference.
np.testing.assert_allclose(res, ref, rtol=1.0e-12, atol=1.0e-12)
print("Verification passed.")
