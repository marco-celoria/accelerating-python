import cupy as cp

shape = (1000, 1000)

# Create a dedicated non-blocking stream for capture and replay
stream = cp.cuda.Stream(non_blocking=True)

# Allocate persistent buffers and enqueue initialization on the stream
with stream:
    x = cp.ones(shape, dtype=cp.float32)
    y = cp.ones(shape, dtype=cp.float32)
    tmp = cp.empty_like(x)
    out = cp.empty_like(x)
    # Warm up the exact operations that will be captured
    cp.sin(x, out=tmp)
    cp.cos(y, out=out)
    cp.add(tmp, out, out=out)

stream.synchronize()

# Capture the static execution sequence using preallocated outputs
with stream:
    stream.begin_capture()
    cp.sin(x, out=tmp)
    cp.cos(y, out=out)
    cp.add(tmp, out, out=out)
    graph = stream.end_capture()

# Replay the graph repeatedly to reduce host launch overhead
for _ in range(100):
    graph.launch(stream=stream)

stream.synchronize()

# Validate the complete result outside stream capture
expected = cp.sin(x) + cp.cos(y)
cp.testing.assert_allclose(out, expected, rtol=1.0e-6, atol=1.0e-7,)
print("CUDA Graph execution completed successfully")
