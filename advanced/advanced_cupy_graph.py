import cupy as cp

# Use a dedicated non-blocking stream for initialization,
# warm-up, capture, replay, and timing.
stream = cp.cuda.Stream(non_blocking=True)

# Allocate persistent device buffers before capture.
# Initialization operations are submitted to this stream.
shape = (1000, 1000)

with stream:
    x = cp.ones(shape, dtype=cp.float32)
    y = cp.ones(shape, dtype=cp.float32)
    tmp = cp.empty_like(x)
    out = cp.empty_like(x)
    # Warm up the exact operations that will be captured.
    # This triggers lazy kernel compilation before capture.
    cp.sin(x, out=tmp)
    cp.cos(y, out=out)
    cp.add(tmp, out, out=out)

stream.synchronize()

# Capture a static sequence using persistent buffers.
with stream:
    stream.begin_capture()
    cp.sin(x, out=tmp)
    cp.cos(y, out=out)
    cp.add(tmp, out, out=out)
    graph = stream.end_capture()

# Warm up graph replay so that the first graph launch is not timed.
graph.launch(stream=stream)
stream.synchronize()

# CUDA Graph replay timing
N = 100
start_graph = cp.cuda.Event()
stop_graph = cp.cuda.Event()
cp.cuda.nvtx.RangePush("Graph", id_color=1)
with stream:
    start_graph.record(stream)
    for _ in range(N):
        graph.launch(stream=stream)
    stop_graph.record(stream)

stop_graph.synchronize()
cp.cuda.nvtx.RangePop()
graph_elapsed_ms = cp.cuda.get_elapsed_time(start_graph, stop_graph)
graph_average_ms = graph_elapsed_ms / N

# Validate the result produced by graph replay before the eager sequence
# writes to the same output buffers.
expected = cp.sin(x) + cp.cos(y)
cp.testing.assert_allclose(out, expected, rtol=1.e-6, atol=1.e-7)

# Ordinary eager-launch timing
start_eager = cp.cuda.Event()
stop_eager = cp.cuda.Event()
cp.cuda.nvtx.RangePush("No Graph", id_color=2)
with stream:
    start_eager.record(stream)
    for _ in range(N):
        cp.sin(x, out=tmp)
        cp.cos(y, out=out)
        cp.add(tmp, out, out=out)
    stop_eager.record(stream)

stop_eager.synchronize()
cp.cuda.nvtx.RangePop()
eager_elapsed_ms = cp.cuda.get_elapsed_time(start_eager, stop_eager)
eager_average_ms = eager_elapsed_ms / N

# Validate the eager result separately.
cp.testing.assert_allclose(out, expected, rtol=1.e-6, atol=1.e-7)
device_time_ratio = eager_elapsed_ms / graph_elapsed_ms
print(f"Array shape                    : {shape}")
print(f"Executions per method          : {N}")
print(f"Graph total device time [ms]   : {graph_elapsed_ms:.6f}")
print(f"Graph average device time [ms] : {graph_average_ms:.6f}")
print(f"Eager total device time [ms]   : {eager_elapsed_ms:.6f}")
print(f"Eager average device time [ms] : {eager_average_ms:.6f}")
print(f"Eager/graph device-time ratio  : {device_time_ratio:.3f}x")
print("Verification passed.")
