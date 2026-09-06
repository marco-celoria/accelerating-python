import numpy as np 
import cupy as cp 

h_x = np.arange(10, dtype=np.float64) 
d_x = cp.asarray(h_x) 
d_y_1 = cp.sin(d_x)**2 
d_y_2 = cp.cos(d_x)**2 
h_r_1 = d_y_1.get() 
h_r_2 = cp.asnumpy(d_y_2) 
ref = h_r_1 + h_r_2
np.testing.assert_allclose(ref, 1.0)
print("Verification passed.")
