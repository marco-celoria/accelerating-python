import cupy as cp

def jacobi_expression(left, right, down, up):
    return 0.25 * (left + right + down + up)

def jacobi_step(d_u, d_v):
    left = d_u[1:-1, :-2]
    right = d_u[1:-1, 2:]
    down = d_u[:-2, 1:-1]
    up = d_u[2:, 1:-1]
    d_v[1:-1, 1:-1] = jacobi_expression(left, right, down, up)


n = 10_000
d_u = cp.random.random((n, n), dtype=cp.float64)
d_v = cp.zeros_like(d_u)

stream = cp.cuda.get_current_stream()
start = cp.cuda.Event()
stop = cp.cuda.Event()

# Warm-up.
jacobi_step(d_u, d_v)
stream.synchronize()

cp.cuda.nvtx.RangePush("Non-fused Jacobi expression", id_color=2)

start.record(stream)
jacobi_step(d_u, d_v)
stop.record(stream)

stop.synchronize()
cp.cuda.nvtx.RangePop()

elapsed_ms = cp.cuda.get_elapsed_time(start, stop)
print(f"Elapsed time: {elapsed_ms:.6f} ms")
