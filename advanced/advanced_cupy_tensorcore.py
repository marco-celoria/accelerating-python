import os
import cupy as cp

fp64_wmma_kernel = cp.RawKernel(
r'''
#include <mma.h>

using namespace nvcuda;

extern "C" __global__
void wmma_fp64(
    const double* __restrict__ A,
    const double* __restrict__ B,
    double* __restrict__ C)
{
    // A single warp (32 threads) computes the 8x8x4 MMA
    wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 8, 8, 4, double> c_frag;
    
    wmma::fill_fragment(c_frag, 0.0);
    wmma::load_matrix_sync(a_frag, A, 4);
    wmma::load_matrix_sync(b_frag, B, 8);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    wmma::store_matrix_sync(C, c_frag, 8, wmma::mem_row_major);
}
''',
    "wmma_fp64",
    options=(
        "--std=c++17",
#        f"-I{os.environ['CUDA_HOME']}/include",
    )
)

# FP64 WMMA uses an 8x8x4 tile configuration.
M = N = 8
K = 4
d_a = cp.random.random((M, K), dtype=cp.float64)
d_b = cp.random.random((K, N), dtype=cp.float64)
d_c = cp.zeros((M, N), dtype=cp.float64)

# Launch the kernel with 1 block of 32 threads (one warp)
fp64_wmma_kernel((1,), (32,), (d_a, d_b, d_c))
cp.cuda.get_current_stream().synchronize()

# Verify the result against CuPy's high-level matrix multiplication
d_r = d_a @ d_b
# Improved assertion
cp.testing.assert_allclose(d_c, d_r, rtol=1e-12, atol=1e-12)
print("Verification passed.")
