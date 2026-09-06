import cupy as cp

long_sin_kernel = cp.ElementwiseKernel(
    in_params="float64 x, int32 iterations",
    out_params="float64 y",
    operation=r"""
        double value = x;
        for (int k = 0; k < iterations; ++k) {
            value = sin(value) + 0.0001 * value;
        }
        y = value;
    """,
    name="long_sin_kernel",
)

long_exp_kernel = cp.ElementwiseKernel(
    in_params="float64 x, int32 iterations",
    out_params="float64 y",
    operation=r"""
        double value = x;
        for (int k = 0; k < iterations; ++k) {
            value = exp(-0.01 * value) + 0.0001 * value;
        }
        y = value;
    """,
    name="long_exp_kernel",
)

long_sqrt_kernel = cp.ElementwiseKernel(
    in_params="float64 x, int32 iterations",
    out_params="float64 y",
    operation=r"""
        double value = x;
        for (int k = 0; k < iterations; ++k) {
            value = sqrt(value) + 0.0001 * value;
        }
        y = value;
    """,
    name="long_sqrt_kernel",
)

n = 10_000
iterations = cp.int32(10_000)
rng = cp.random.default_rng(seed=42)
x = rng.random(n, dtype=cp.float64)

# Blocking custom streams.
stream1 = cp.cuda.Stream(non_blocking=False)
stream2 = cp.cuda.Stream(non_blocking=False)
stream3 = cp.cuda.Stream(non_blocking=False)

# Completion event for the operation producing y2.
y2_ready = cp.cuda.Event(disable_timing=True)

# Trigger JIT compilation before the profiled execution.
warmup_x = x[:1024]
long_sin_kernel(warmup_x, iterations)
long_exp_kernel(warmup_x, iterations)
long_sqrt_kernel(warmup_x, iterations)
cp.cuda.Stream.null.synchronize()

# The blocking custom streams wait for this preceding NULL-stream work.
cp.cuda.nvtx.RangePush("Submit Operation 0 to Default Stream", id_color=1)
y0 = long_sqrt_kernel(x, iterations)
cp.cuda.nvtx.RangePop()

cp.cuda.nvtx.RangePush("Submit Operation 1 to Stream 1", id_color=2)
with stream1:
    y1 = long_sin_kernel(x, iterations)
cp.cuda.nvtx.RangePop()

cp.cuda.nvtx.RangePush("Submit Operation 2 to Stream 2", id_color=3,)
with stream2:
    y2 = long_exp_kernel(x, iterations)
    y2_ready.record()
cp.cuda.nvtx.RangePop()

# Operation 3 consumes y2.
stream3.wait_event(y2_ready)

cp.cuda.nvtx.RangePush("Submit Operation 3 to Stream 3", id_color=4,)
with stream3:
    y3 = long_sqrt_kernel(y2, iterations)
cp.cuda.nvtx.RangePop()

# Wait for the independent branch.
stream1.synchronize()

# This also waits transitively for Operation 2.
stream3.synchronize()

print("Execution completed")
