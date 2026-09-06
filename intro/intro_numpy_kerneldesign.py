import numpy as np

def jacobi_point_update(u, unew, i, j, n):
    # Defensive check retained to mirror the CUDA implementation.
    # It is redundant for the loop bounds used below.
    if i <= 0 or i >= n - 1 or j <= 0 or j >= n - 1:
        return
    idx = i * n + j
    left  = u[idx - 1]
    right = u[idx + 1]
    down  = u[idx - n]
    up    = u[idx + n]
    unew[idx] = 0.25 * (left + right + down + up)


n = 1_000

u = np.random.random(n * n).astype(np.float64)
unew = np.zeros_like(u)

# The loop bounds contain only physical-interior points.
for i in range(1, n - 1):
    for j in range(1, n - 1):
        jacobi_point_update(u, unew, i, j, n)

print("Execution complete.")
