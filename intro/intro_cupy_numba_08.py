import cupy as cp

# Compute c = a + b element by element.
array_add = cp.ElementwiseKernel(
    in_params="T a, T b",
    out_params="T c",
    operation="c = a + b;",
    name="array_add",
)

# Compute sqrt(sum(x[i] * x[i])) using FP64 accumulation.
frobenius_norm = cp.ReductionKernel(
    in_params="T x",
    out_params="float64 norm",
    # Explicitly cast mapped values and accumulate in FP64.
    map_expr="(double)x * (double)x", 
    reduce_expr="a + b",
    post_map_expr="norm = sqrt(a)",
    identity="0.0",
    reduce_type="double",
    name="frobenius_norm",
)

rows = 4096
cols = 4096
d_a = cp.random.random((rows, cols), dtype=cp.float64)
d_b = cp.random.random((rows, cols), dtype=cp.float64)
d_c = cp.empty_like(d_a)

# Warm up both kernels to trigger compilation and caching.
array_add(d_a, d_b, d_c)
frobenius_norm(d_c)
cp.cuda.get_current_stream().synchronize()

# Profile the cached kernels.
cp.cuda.nvtx.RangePush("Elementwise addition and Frobenius norm")
array_add(d_a, d_b, d_c)
norm = frobenius_norm(d_c)
cp.cuda.get_current_stream().synchronize()
cp.cuda.nvtx.RangePop()

# Verify the results outside the profiled region.
d_r = d_a + d_b
norm_r = cp.linalg.norm(d_r)
# Improved Assertions
cp.testing.assert_allclose(d_c, d_r, rtol=1.0e-12, atol=1.0e-12)
cp.testing.assert_allclose(norm, norm_r, rtol=1.0e-12, atol=1.0e-12)
print("Verification passed.")
