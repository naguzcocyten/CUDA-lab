import numpy as np
from numba import cuda
import math
import time

@numba.cuda.jit
def matrix_scale_kernel(mat, scalar, out):
    """
    Scale every element: out[row, col] = mat[row, col] * scalar
    """
    row, col = cuda.grid(2)

    if row < out.shape[0] and col < out.shape[1]:
        out[row, col] = mat[row, col] * scalar


def main():
    rows_large, cols_large = 4096, 4096
    mat = np.random.randn(rows_large, cols_large).astype(np.float32)
    out = np.zeros_like(mat)
    scalar = 2.5
    d_mat = cuda.to_device(mat)
    d_out = cuda.to_device(out)

    threads_per_block = (16, 16)
    blocks_per_grid_x = math.ceil(rows_large / threads_per_block[0])
    blocks_per_grid_y = math.ceil(cols_large / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # Warmup
    matrix_scale_kernel[blocks_per_grid, threads_per_block](d_mat, scalar, d_out)
    cuda.synchronize()

    # GPU timing
    start = time.time()
    matrix_scale_kernel[blocks_per_grid, threads_per_block](d_mat, scalar, d_out)
    cuda.synchronize()
    gpu_time = (time.time() - start) * 1000

    result = d_out.copy_to_host()

    # CPU timing
    cpu_start = time.time()
    expected = mat * scalar
    cpu_time = (time.time() - cpu_start) * 1000

    print(f"GPU kernel time: {gpu_time:.3f} ms")
    print(f"CPU NumPy time: {cpu_time:.3f} ms")
    print(f"Speedup: {cpu_time / gpu_time:.2f}x")
    print("Correct:", np.allclose(result, expected))

if __name__ == "__main__":
    main()
