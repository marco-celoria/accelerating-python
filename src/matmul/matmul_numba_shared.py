from numba import cuda, float32, float64
import numpy as np
import math
import nvtx
import argparse
from time import perf_counter


# Controls threads per block and shared memory usage.
# The computation will be done on blocks of TPBxTPB elements.
# TPB should not be larger than 32 in this example
TPB = 16


@cuda.jit
def fast_matmul_float64(A, B, C):
    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float64)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float64)
    b_c = cuda.blockIdx.x
    b_r = cuda.blockIdx.y
    t_c = cuda.threadIdx.x
    t_r = cuda.threadIdx.y
    col = b_c * TPB + t_c
    row = b_r * TPB + t_r
    # The dot product is chunked into dot products of TPB-long vectors.
    Cvalue = float64(0.)
    num_tiles = (A.shape[1] + TPB - 1) // TPB
    for tile in range(num_tiles):
        col_A = tile * TPB + t_c
        row_B = tile * TPB + t_r
        # Preload data into shared memory
        if row < A.shape[0] and col_A < A.shape[1]:
            sA[t_r, t_c]= A[row, col_A]  # copy A to shared mem
        else:
            sA[t_r,t_c]= float64(0.)
        if row_B < B.shape[0] and col < B.shape[1]:
            sB[t_r,t_c]= B[row_B, col]  # copy B to shared mem
        else:
            sB[t_r,t_c]= float64(0.)
        # Wait until all threads finish preloading
        cuda.syncthreads()
        # Computes partial product on the shared memory
        for k in range(TPB):
            Cvalue += sA[t_r, k] * sB[k, t_c]
        # Wait until all threads finish computing
        cuda.syncthreads()
    if row < A.shape[0] and col < B.shape[1]:
        C[row, col] = Cvalue

#@cuda.jit
#def fast_matmul_float32(A, B, C):
#    # Define an array in the shared memory
#    # The size and type of the arrays must be known at compile time
#    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
#    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
#    x, y = cuda.grid(2)
#    tx = cuda.threadIdx.x
#    ty = cuda.threadIdx.y
#    bpg = cuda.gridDim.x    # blocks per grid
#    # Each thread computes one element in the checkult matrix.
#    # The dot product is chunked into dot products of TPB-long vectors.
#    tmp = float32(0.)
#    for i in range(bpg):
#        # Preload data into shared memory
#        sA[ty, tx] = 0
#        sB[ty, tx] = 0
#        if y < A.shape[0] and (tx+i*TPB) < A.shape[1]:
#          sA[ty, tx] = A[y, tx + i * TPB]
#        if x < B.shape[1] and (ty+i*TPB) < B.shape[0]:
#          sB[ty, tx] = B[ty + i * TPB, x]
#        # Wait until all threads finish preloading
#        cuda.syncthreads()
#        # Computes partial product on the shared memory
#        for j in range(TPB):
#            tmp += sA[ty, j] * sB[j, tx]
#        # Wait until all threads finish computing
#        cuda.syncthreads()
#    if y < C.shape[0] and x < C.shape[1]:
#        C[y, x] = tmp
       
@cuda.jit
def fast_matmul_float32(A, B, C):
    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    b_c = cuda.blockIdx.x
    b_r = cuda.blockIdx.y
    t_c = cuda.threadIdx.x
    t_r = cuda.threadIdx.y
    col = b_c * TPB + t_c
    row = b_r * TPB + t_r
    # The dot product is chunked into dot products of TPB-long vectors.
    Cvalue = float32(0.)
    num_tiles = (A.shape[1] + TPB - 1) // TPB
    for tile in range(num_tiles):
        col_A = tile * TPB + t_c
        row_B = tile * TPB + t_r
        # Preload data into shared memory
        if row < A.shape[0] and col_A < A.shape[1]:
            sA[t_r, t_c]= A[row, col_A]  # copy A to shared mem
        else:
            sA[t_r,t_c]= float32(0.)
        if row_B < B.shape[0] and col < B.shape[1]:
            sB[t_r,t_c]= B[row_B, col]  # copy B to shared mem
        else:
            sB[t_r,t_c]= float32(0.)
        # Wait until all threads finish preloading
        cuda.syncthreads()
        # Computes partial product on the shared memory
        for k in range(TPB):
            Cvalue += sA[t_r, k] * sB[k, t_c]
        # Wait until all threads finish computing
        cuda.syncthreads()
    if row < A.shape[0] and col < B.shape[1]:
        C[row, col] = Cvalue


def matmul(a, b, c, iters, kernel):
    a_h = a.copy_to_host()
    b_h = b.copy_to_host()
    threadsperblock = (TPB, TPB)
    grid_y_max = a.shape[0] #max(a.shape[0],b.shape[0])
    grid_x_max = b.shape[1] #max(a.shape[1],b.shape[1])
    blockspergrid_x = math.ceil(grid_x_max / threadsperblock[0])
    blockspergrid_y = math.ceil(grid_y_max / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    print(f"blockspergrid = {blockspergrid} - threadsperblock = {threadsperblock}")
    cuda.synchronize()
    timings_gpu = np.empty(iters)
    timings_cpu = np.empty(iters)
    for i in range(iters):
        cuda.synchronize()
        tic = perf_counter()
        with nvtx.annotate("Kernels", color="green"):
            kernel[blockspergrid, threadsperblock](a, b, c)
        cuda.synchronize()
        toc = perf_counter()
        timings_gpu[i] = toc - tic
        tic = perf_counter()
        check_h = a_h @ b_h
        toc = perf_counter()
        timings_cpu[i] = toc - tic
        c_h = c.copy_to_host()
        print("Check:", np.allclose(c_h, check_h), np.linalg.norm(c_h - check_h), np.max(np.abs(c_h - check_h)))
    print(f"GPU Elapsed time: {timings_gpu.mean()} s")
    print(f"CPU Elapsed time: {timings_cpu.mean()} s")
    print(f"Speedup: {timings_cpu.mean() / timings_gpu.mean():.2f}x")
    gops = a.shape[1] * a.shape[0] * (2. * b.shape[1] - 1.) * 1.e-9
    gpu_flops_s= gops/ timings_gpu.mean()
    cpu_flops_s= gops/ timings_cpu.mean()
    print(f"GPU GFLOPS/s: {gpu_flops_s:.4f}")
    print(f"CPU GFLOPS/s: {cpu_flops_s:.4f}")
#
#        if a.shape[0] < 250 and b.shape[1] < 250 and a.shape[1] < 250:
#            a_h = a.copy_to_host()
#            b_h = b.copy_to_host()
#            c_h = c.copy_to_host()
#            check_h = a_h @ b_h
#            print("Check:", np.allclose(c_h, check_h), np.linalg.norm(c_h - check_h), np.max(np.abs(c_h - check_h)))
#    print(f"Elapsed time: {timings.mean()} ± {timings.std()} s")
#    gops = a.shape[1] * a.shape[0] * (2. * b.shape[1] - 1.) * 1.e-9
#    gpu_flops_s= gops/ timings.mean()
#    print(f"GPU GFLOPS/s: {gpu_flops_s:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("n",     nargs='?', type=int, help="n", default=99)
    parser.add_argument("m",     nargs='?', type=int, help="m", default=123)
    parser.add_argument("l",     nargs='?', type=int, help="l", default=78)
    parser.add_argument("iters", nargs='?', type=int, help="iterations", default=1)
    args = parser.parse_args()
    n = args.n
    m = args.m
    l = args.l
    iters = args.iters
    print(f"{n}x{l} {l}x{m} - {n} x {m} - {iters}")
    z_h = np.zeros((n, m)).astype(np.float64)
    x_h = np.random.randn(n, l).astype(np.float64)
    y_h = np.random.randn(l, m).astype(np.float64)

    with nvtx.annotate("H2D Memory", color="yellow"):
      x_d = cuda.to_device(x_h)
      y_d = cuda.to_device(y_h)
      z_d = cuda.to_device(z_h)

    if x_h.dtype == np.float32:
        matmul(x_d, y_d, z_d, iters, fast_matmul_float32)
    elif x_h.dtype == np.float64:
        matmul(x_d, y_d, z_d, iters, fast_matmul_float64)
    else:
        raise TypeError(f"Type must be either float32 or float64")
  
    

if __name__ == "__main__":
  main()

