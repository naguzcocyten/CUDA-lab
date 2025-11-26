# ex1_vector_add.py
import numpy as np
from numba import cuda
import math
import time

@cuda.jit
def vector_add_kernel(a, b, c):
    """
    Each thread computes one element: c[i] = a[i] + b[i]
    """
    # Compute global thread index
    idx = cuda.grid(1)

    # Boundary check
    if idx < c.size:
        c[idx] = a[idx] + b[idx]

def main():
    N_large = 10_000_000
    a = np.random.randn(N_large).astype(np.float32)
    b = np.random.randn(N_large).astype(np.float32)
    c = np.zeros(N_large, dtype=np.float32)

    d_a = cuda.to_device(a)
    d_b = cuda.to_device(b)
    d_c = cuda.to_device(c)

    threads_per_block = 256
    blocks_per_grid = math.ceil(N_large / threads_per_block)

    # Warmup
    vector_add_kernel[blocks_per_grid, threads_per_block](d_a, d_b, d_c)
    cuda.synchronize()

    # GPU timing
    start = time.time()
    vector_add_kernel[blocks_per_grid, threads_per_block](d_a, d_b, d_c)
    cuda.synchronize()
    gpu_time = (time.time() - start) * 1000

    result = d_c.copy_to_host()

    # CPU timing
    cpu_start = time.time()
    expected = a + b
    cpu_time = (time.time() - cpu_start) * 1000

    print(f"GPU kernel time: {gpu_time:.3f} ms")
    print(f"CPU NumPy time: {cpu_time:.3f} ms")
    print(f"Speedup: {cpu_time / gpu_time:.2f}x")
    print("Correct:", np.allclose(result, expected))

if __name__ == "__main__":
    main()
