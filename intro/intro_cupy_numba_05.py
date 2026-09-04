import cupy as cp

n = 1_000
m = 100
l = 10
# Element-wise operations and reduction 
d_x = cp.linspace(0.0, 1.0, n * m)
d_y = cp.sin(d_x)
d_z2 = 1.0 - d_y**2 
d_total = d_z2.sum()

# Sorting and reshaping 
d_A = cp.sort(d_y).reshape(n, m)

# Computes the matrix Frobenius norm ||d_A||_F on the GPU
d_norm = cp.linalg.norm(d_A)

# Matrix multiplication 
d_B = cp.random.random((m, l))
d_C = d_A @ d_B
cp.testing.assert_allclose(d_total, (cp.cos(d_x)**2).sum(), rtol=1e-7)
assert d_C.shape == (n, l), f"Expected shape {(n, l)}, got {d_C.shape}"
print("Verification passed.")
