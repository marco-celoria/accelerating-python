import cupy as cp

@cp.fuse(kernel_name="fused_jacobi_expression")
def fused_jacobi_expression(left, right, down, up):
    return 0.25 * (left + right + down + up)

def jacobi_step_fused(d_u, d_v):
    left = d_u[1:-1, :-2]
    right = d_u[1:-1, 2:]
    down = d_u[:-2, 1:-1]
    up = d_u[2:, 1:-1]
    d_v[1:-1, 1:-1] = fused_jacobi_expression(left, right, down, up)


n = 10_000
d_u = cp.random.random((n, n), dtype=cp.float64)
d_v = cp.zeros_like(d_u)

stream = cp.cuda.get_current_stream()
start = cp.cuda.Event()
stop = cp.cuda.Event()

# Warm up and compile the fused kernel.
jacobi_step_fused(d_u, d_v)
stream.synchronize()

cp.cuda.nvtx.RangePush("Fused Jacobi expression", id_color=2,)

start.record(stream)
jacobi_step_fused(d_u, d_v)
stop.record(stream)

stop.synchronize()
cp.cuda.nvtx.RangePop()

elapsed_ms = cp.cuda.get_elapsed_time(start, stop)
print(f"Elapsed time: {elapsed_ms:.6f} ms")
d_r = cp.zeros_like(d_u)
d_r[1:-1, 1:-1] = 0.25 * (d_u[1:-1, :-2] + d_u[1:-1, 2:] + d_u[:-2, 1:-1] + d_u[2:, 1:-1])
cp.testing.assert_allclose(d_v, d_r, rtol=1.0e-12, atol=1.0e-12,)
print("Verification passed.")
