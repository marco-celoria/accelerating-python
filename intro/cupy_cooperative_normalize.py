import cupy as cp
import numpy as np

WARP_SIZE = 32
THREADS_PER_BLOCK = 256

if THREADS_PER_BLOCK % WARP_SIZE != 0:
    raise ValueError("THREADS_PER_BLOCK must be a multiple of 32")

if THREADS_PER_BLOCK > 1024:
    raise ValueError("THREADS_PER_BLOCK must not exceed 1024")

kernel_source = r"""
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

extern "C" __global__
void cooperative_normalize(
    const double* __restrict__ x,
    double* __restrict__ y,
    double* __restrict__ global_sum,
    const unsigned long long n)
{
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    const unsigned int lane_id = warp.thread_rank();
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int number_of_warps = blockDim.x / 32;

    const unsigned long long global_thread =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long grid_stride =
        static_cast<unsigned long long>(blockDim.x) * gridDim.x;

    // Reinterpret the scalar arrays as CUDA double2 vectors. 
    const double2* x_vec = reinterpret_cast<const double2*>(x);
    double2* y_vec = reinterpret_cast<double2*>(y);
    const unsigned long long n_vec = n / 2;

    // Stage 1: process two adjacent doubles per iteration.
    double thread_sum = 0.0;
    for (unsigned long long i = global_thread; i < n_vec; i += grid_stride) {
        const double2 v = x_vec[i];
        thread_sum += v.x + v.y;
    }
    // If n is odd, thread zero processes the final scalar.
    if (global_thread == 0 && (n % 2 != 0)) {
        thread_sum += x[n - 1];
    }

    // Stage 2: Warp reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        thread_sum += warp.shfl_down(thread_sum, offset);
    }

    __shared__ double warp_sums[32];
    if (lane_id == 0) {
        warp_sums[warp_id] = thread_sum;
    }
    block.sync();

    // Stage 3: Block reduction & single atomic accumulation
    if (warp_id == 0) {
        double block_sum = 0.0;
        if (lane_id < number_of_warps) {
            block_sum = warp_sums[lane_id];
        }
        for (int offset = 16; offset > 0; offset /= 2) {
            block_sum += warp.shfl_down(block_sum, offset);
        }
        if (lane_id == 0) {
            atomicAdd(global_sum, block_sum);
        }
    }

    // Stage 4: Wait until all blocks have updated global_sum
    grid.sync();

    const double sum = global_sum[0];

    // Stage 5: normalize two adjacent doubles per iteration.
    for (unsigned long long i = global_thread; i < n_vec; i += grid_stride) {
        double2 v = x_vec[i];
        double2 res;
        res.x = v.x / sum;
        res.y = v.y / sum;
        y_vec[i] = res; // Store two adjacent doubles through a double2 access.
    }
    
    if (global_thread == 0 && (n % 2 != 0)) {
        y[n - 1] = x[n - 1] / sum;
    }
}
"""

cooperative_normalize = cp.RawKernel(
    kernel_source,
    "cooperative_normalize",
    options=("--std=c++17",),
    enable_cooperative_groups=True,
)

# Compile the kernel before querying its occupancy.
cooperative_normalize.compile()

device = cp.cuda.Device()
device_attributes = device.attributes

if not device_attributes.get("CooperativeLaunch", 0):
    raise RuntimeError(
        "The selected CUDA device does not support cooperative launches."
    )

if THREADS_PER_BLOCK > cooperative_normalize.max_threads_per_block:
    raise ValueError(
        f"THREADS_PER_BLOCK={THREADS_PER_BLOCK} exceeds the compiled "
        f"kernel limit of {cooperative_normalize.max_threads_per_block}"
    )

max_blocks_per_sm = (
    cp.cuda.driver.occupancyMaxActiveBlocksPerMultiprocessor(
        cooperative_normalize.kernel.ptr,
        THREADS_PER_BLOCK,
        0,
    )
)

if max_blocks_per_sm <= 0:
    raise RuntimeError(
        "The occupancy API reported no active blocks per SM."
    )

number_of_sms = device_attributes["MultiProcessorCount"]
number_of_blocks = number_of_sms * max_blocks_per_sm

# Warm-up launch
warmup_n = THREADS_PER_BLOCK
warmup_x = cp.ones(warmup_n, dtype=cp.float64)
warmup_y = cp.empty_like(warmup_x)

# The atomic accumulator must be zero before every launch.
warmup_sum = cp.zeros(1, dtype=cp.float64)

cooperative_normalize(
    (1,),
    (THREADS_PER_BLOCK,),
    (
        warmup_x,
        warmup_y,
        warmup_sum,
        np.uint64(warmup_n),
    ),
)

cp.cuda.get_current_stream().synchronize()

# Profile launch
n = 1_000_000_001
x = cp.random.random(n, dtype=cp.float64)
y = cp.empty_like(x)

if not x.flags.c_contiguous or not y.flags.c_contiguous:
    raise ValueError("x and y must be C-contiguous")

if x.data.ptr % 16 != 0 or y.data.ptr % 16 != 0:
    raise ValueError(
        "x and y must be 16-byte aligned for double2 access"
    )

# The atomic accumulator must be zero before every launch.
global_sum = cp.zeros(1, dtype=cp.float64)

cp.cuda.nvtx.RangePush(
    "CuPy Cooperative Normalize",
    id_color=1,
)

cooperative_normalize(
    (number_of_blocks,),
    (THREADS_PER_BLOCK,),
    (
        x,
        y,
        global_sum,
        np.uint64(n),
    ),
)

cp.cuda.get_current_stream().synchronize()
cp.cuda.nvtx.RangePop()

# Verification
computed_sum = global_sum[0].item()

cp.cuda.nvtx.RangePush(
    "CuPy Normalization Reference",
    id_color=2,
)
reference_sum = cp.sum(x, dtype=cp.float64).item()
reference_y = x / reference_sum
cp.cuda.nvtx.RangePop()

normalized_sum = cp.sum(y, dtype=cp.float64).item()

absolute_error = abs(computed_sum - reference_sum)
relative_error = absolute_error / abs(reference_sum)

print("Execution blocks:     ", number_of_blocks)
print("Total threads:        ", number_of_blocks * THREADS_PER_BLOCK)
print("Blocks per SM:        ", max_blocks_per_sm)
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
