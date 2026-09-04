import cupy as cp

d_x = cp.linspace(0.0, 1.0, 1_000_000)
d_y = cp.sin(d_x * cp.pi / 2.0)**2
cp.testing.assert_allclose(d_y[-1], 1.0, rtol=1e-15, atol=1e-15)
print("Verification passed.")
