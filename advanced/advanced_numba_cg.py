import cupy as cp
from numba import cuda, float64

WARP_SIZE = 32
FULL_MASK = 0xFFFFFFFF
MAX_WARPS_PER_BLOCK = 32
THREADS_PER_BLOCK = 256

if not 1 <= THREADS_PER_BLOCK <= 1024:
    raise ValueError("THREADS_PER_BLOCK must be between 1 and 1024")

if THREADS_PER_BLOCK % WARP_SIZE != 0:
    raise ValueError("THREADS_PER_BLOCK must be a multiple of 32")

if THREADS_PER_BLOCK // WARP_SIZE > MAX_WARPS_PER_BLOCK:
    raise ValueError("The shared warp_sums array is too small for this block size")

@cuda.jit
def cooperative_normalize(x, y, global_sum):
    """
    Compute:
        y[i] = x[i] / sum(x)
    in one cooperative kernel.
    The kernel uses:
      1. cuda.shfl_down_sync for warp-level reduction.
      2. Shared memory for block-level reduction.
      3. cuda.atomic.add for one contribution per block.
      4. cuda.cg.this_grid().sync() for a grid-wide barrier.
    """

    grid = cuda.cg.this_grid()
    warp_sums = cuda.shared.array(shape=MAX_WARPS_PER_BLOCK, dtype=float64)
    thread_id = cuda.threadIdx.x
    lane_id = thread_id % WARP_SIZE
    warp_id = thread_id // WARP_SIZE
    number_of_warps = cuda.blockDim.x // WARP_SIZE
    global_thread = cuda.grid(1)
    grid_stride = cuda.gridsize(1)

    # Stage 1: grid-stride thread-local accumulation.
    thread_sum = float64(0.0)
    for i in range(global_thread, x.size, grid_stride):
        thread_sum += x[i]

    # Stage 2: warp-level reduction.
    offset = WARP_SIZE // 2
    while offset > 0:
        other = cuda.shfl_down_sync(FULL_MASK, thread_sum, offset)
        thread_sum += other
        offset //= 2

    # Lane zero writes one result per warp.
    if lane_id == 0:
        warp_sums[warp_id] = thread_sum

    # Numba does not currently expose a thread-block Cooperative
    # Groups object equivalent to cg::this_thread_block().
    # cuda.syncthreads() provides the required block barrier.
    cuda.syncthreads()

    # Stage 3: the first warp reduces the warp partial sums.
    if warp_id == 0:
        block_sum = float64(0.0)
        if lane_id < number_of_warps:
            block_sum = warp_sums[lane_id]
        offset = WARP_SIZE // 2
        while offset > 0:
            other = cuda.shfl_down_sync(FULL_MASK, block_sum, offset)
            block_sum += other
            offset //= 2

        # One global atomic addition per block.
        if lane_id == 0:
            cuda.atomic.add(global_sum, 0, block_sum)

    # Stage 4: wait until all blocks have updated global_sum.
    grid.sync()

    # Stage 5: normalize using the completed global sum.
    sum_value = global_sum[0]
    for i in range(global_thread, x.size, grid_stride):
        y[i] = x[i] / sum_value


device = cuda.get_current_device()
number_of_sms = device.MULTIPROCESSOR_COUNT

if number_of_sms <= 0:
    raise RuntimeError(f"Invalid SM count reported by CUDA: {number_of_sms}")

# Warm-up and JIT compilation.
warmup_n = 100 * THREADS_PER_BLOCK
warmup_x = cp.ones(warmup_n, dtype=cp.float64)
warmup_y = cp.empty_like(warmup_x)
wormup_reference = cp.empty_like(warmup_x)

# The atomic accumulator must be zero before every launch.
warmup_sum = cp.zeros(1, dtype=cp.float64)
# Ensure CuPy initialization is complete before the Numba launch.
cp.cuda.get_current_stream().synchronize()
cooperative_normalize[100, THREADS_PER_BLOCK](warmup_x, warmup_y, warmup_sum)

cp_sum = cp.sum(warmup_x, dtype=cp.float64)
cp.divide(warmup_x, cp_sum, out=wormup_reference)
cuda.synchronize()

# WORKAROUND:
# Verified on the current GPU and software stack for:
#   THREADS_PER_BLOCK = 256
#   numba-cuda = 0.30.4
#   cuda-bindings = 12.9.7
#   cuda-core = 0.6.0
#
# On this stack, launches above five blocks per SM fail with
# CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE despite the larger value
# returned by max_cooperative_grid_blocks().

VERIFIED_COOPERATIVE_BLOCKS_PER_SM = 5
number_of_overloads = len(cooperative_normalize.overloads)

if number_of_overloads != 1:
    raise RuntimeError(f"Expected exactly one compiled kernel specialization, found {number_of_overloads}.")

kernel_overload = next(iter(cooperative_normalize.overloads.values()))
reported_maximum_blocks = (
    kernel_overload.max_cooperative_grid_blocks(
        THREADS_PER_BLOCK,
        0,
    )
)

if reported_maximum_blocks <= 0:
    raise RuntimeError("The occupancy API reported no valid cooperative grid blocks.")

reported_blocks_per_sm = reported_maximum_blocks // number_of_sms

if reported_blocks_per_sm <= 0:
    raise RuntimeError(f"The reported cooperative grid limit is smaller than the SM count: {reported_maximum_blocks} blocks for {number_of_sms} SMs.")

selected_blocks_per_sm = min(VERIFIED_COOPERATIVE_BLOCKS_PER_SM, reported_blocks_per_sm,)
number_of_blocks = selected_blocks_per_sm * number_of_sms
if number_of_blocks > reported_maximum_blocks:
    raise RuntimeError(f"Selected grid size {number_of_blocks} exceeds the reported limit {reported_maximum_blocks}")

print("SM count:                            ", number_of_sms)
print("Reported cooperative blocks:         ", reported_maximum_blocks)
print("Reported blocks per SM:              ", reported_blocks_per_sm)
print("Verified cooperative blocks per SM:  ", VERIFIED_COOPERATIVE_BLOCKS_PER_SM)
print("Selected blocks per SM:              ", selected_blocks_per_sm)
print("Blocks selected for launch:          ", number_of_blocks)

n = 2_000_000_001
x = cp.random.random(n, dtype=cp.float64)
y = cp.empty_like(x)
reference_y = cp.empty_like(x)
global_sum = cp.zeros(1, dtype=cp.float64)
cp.cuda.get_current_stream().synchronize()
cp.cuda.nvtx.RangePush("Numba cooperative normalize", id_color=1,)

try:
    cooperative_normalize[number_of_blocks, THREADS_PER_BLOCK,](x, y, global_sum,)
    cuda.synchronize()
finally:
    cp.cuda.nvtx.RangePop()

cp.cuda.nvtx.RangePush("CuPy Normalization Reference", id_color=2)
reference_sum = cp.sum(x, dtype=cp.float64)
cp.divide(x, reference_sum, out=reference_y)
cuda.synchronize()
cp.cuda.nvtx.RangePop()

# Verification
computed_sum = global_sum[0].item()

reference_sum = reference_sum.item()
normalized_sum = cp.sum(y, dtype=cp.float64).item()

absolute_error = abs(computed_sum - reference_sum)
relative_error = absolute_error / abs(reference_sum)

print("Execution blocks:     ", number_of_blocks)
print("Total threads:        ", number_of_blocks * THREADS_PER_BLOCK)
print("Blocks per SM:        ", selected_blocks_per_sm)
print("Computed FP64 sum:    ", computed_sum)
print("Reference FP64 sum:   ", reference_sum)
print("Absolute sum error:   ", absolute_error)
print("Relative sum error:   ", relative_error)
print("Normalized sum:       ", normalized_sum)

cp.testing.assert_allclose(
    computed_sum,
    reference_sum,
    rtol=1.0e-11,
    atol=1.0e-5,
)

cp.testing.assert_allclose(
    normalized_sum,
    1.0,
    rtol=1.0e-11,
    atol=1.0e-10,
)

cp.testing.assert_allclose(
    y,
    reference_y,
    rtol=1.0e-11,
    atol=1.0e-10,
)

print("Verification passed.")

