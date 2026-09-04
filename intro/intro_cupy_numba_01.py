import numpy as np

h_x = np.linspace(0.0, 1.0, 1_000_000)
h_y = np.sin(h_x * np.pi / 2.0)**2
np.testing.assert_allclose(h_y[-1], 1.0, rtol=1e-15, atol=1e-15)
print("Verification passed.")
