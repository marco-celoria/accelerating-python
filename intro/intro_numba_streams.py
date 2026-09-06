import math
import numpy as np
import cupy as cp
from numba import cuda

@cuda.jit
def long_sin_kernel(x, y, iterations):
    i = cuda.grid(1)
    if i < x.size:
        value = x[i]
        for _ in range(iterations):
            value = math.sin(value) + 0.0001 * value
        y[i] = value

@cuda.jit
def long_exp_kernel(x, y, iterations):
    i = cuda.grid(1)
    if i < x.size:
        value = x[i]
        for _ in range(iterations):
            value = math.exp(-0.01 * value) + 0.0001 * value
        y[i] = value

@cuda.jit
def long_sqrt_kernel(x, y, iterations):
    i = cuda.grid(1)
    if i < x.size:
        value = x[i]
        for _ in range(iterations):
            value = math.sqrt(value) + 0.0001 * value
        y[i] = value

# Problem configuration
n = 10_000
iterations = np.int32(10_000)
threads_per_block = 256
blocks_per_grid = (n + threads_per_block - 1) // threads_per_block

# Initialize the input
rng = np.random.default_rng(seed=42)
h_x = rng.random(n).astype(np.float64)
d_x = cuda.to_device(h_x)
d_y0 = cuda.device_array_like(d_x)
d_y1 = cuda.device_array_like(d_x)
d_y2 = cuda.device_array_like(d_x)
d_y3 = cuda.device_array_like(d_x)

# Create CUDA streams and the dependency event
# Explicitly select the legacy default stream so its synchronization behavior
# with ordinary CUDA streams is unambiguous.
default_stream = cuda.legacy_default_stream()

# cuda.stream() creates ordinary CUDA streams. These retain legacy
# synchronization behavior with the legacy default stream.
stream1 = cuda.stream()
stream2 = cuda.stream()
stream3 = cuda.stream()

# Timing is unnecessary because the event is used only for synchronization.
y2_ready = cuda.event(timing=False)

# Warm up the kernels
# Use 1,024 elements to trigger Numba JIT compilation before profiling.
warmup_elements = 1_024
warmup_blocks = (warmup_elements + threads_per_block - 1) // threads_per_block
d_warmup_x = d_x[:warmup_elements]
d_warmup_y = cuda.device_array(warmup_elements, dtype=np.float64,)
long_sin_kernel[ warmup_blocks, threads_per_block, default_stream,](d_warmup_x, d_warmup_y, iterations)
long_exp_kernel[ warmup_blocks, threads_per_block, default_stream,](d_warmup_x, d_warmup_y, iterations)
long_sqrt_kernel[warmup_blocks, threads_per_block, default_stream,](d_warmup_x, d_warmup_y, iterations)
default_stream.synchronize()

# Submit the profiled operations
cp.cuda.nvtx.RangePush("Submit Operation 0 to Default Stream", id_color=1)
long_sqrt_kernel[blocks_per_grid, threads_per_block, default_stream](d_x, d_y0, iterations)
cp.cuda.nvtx.RangePop()


cp.cuda.nvtx.RangePush("Submit Operation 1 to Stream 1", id_color=2)
long_sin_kernel[blocks_per_grid, threads_per_block, stream1](d_x, d_y1, iterations)
cp.cuda.nvtx.RangePop()

cp.cuda.nvtx.RangePush("Submit Operation 2 to Stream 2", id_color=3)
long_exp_kernel[blocks_per_grid, threads_per_block, stream2](d_x, d_y2, iterations)
# Record completion of the kernel producing d_y2.
y2_ready.record(stream2)
cp.cuda.nvtx.RangePop()

# Operation 3 consumes d_y2 and must therefore wait for Operation 2.
y2_ready.wait(stream3)

cp.cuda.nvtx.RangePush("Submit Operation 3 to Stream 3", id_color=4)
long_sqrt_kernel[blocks_per_grid, threads_per_block, stream3](d_y2, d_y3, iterations)
cp.cuda.nvtx.RangePop()

# Wait for the independent branch.
stream1.synchronize()
# This waits for Operation 3 and, transitively, for Operation 2.
stream3.synchronize()
print("Execution completed")
