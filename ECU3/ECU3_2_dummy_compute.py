import numpy as np
from numba import cuda
import math
import time

@numba.cuda.jit
def dummy_compute_kernel(a, b, c):
    """
    Simple compute to measure timing: c[i] = sqrt(a[i]^2 + b[i]^2)
    """
    idx = cuda.grid(1)
    if idx < c.size:
        c[idx] = math.sqrt(a[idx]**2 + b[idx]**2)

def main():
    N = 10_000_000   # 1M elements
    a = np.random.randn(N).astype(np.float32)
    b = np.random.randn(N).astype(np.float32)
    c = np.zeros(N, dtype=np.float32)

    # Device arrays
    d_a = cuda.to_device(a)
    d_b = cuda.to_device(b)
    d_c = cuda.to_device(c)

    threads_per_block = 256
    blocks_per_grid = math.ceil(N / threads_per_block)
    # Warmup (first launch can be slower due to JIT compilation)
    dummy_compute_kernel[blocks_per_grid, threads_per_block](d_a, d_b, d_c)
    cuda.synchronize()

    # Timed run
    start = time.time()
    dummy_compute_kernel[blocks_per_grid, threads_per_block](d_a, d_b, d_c)
    cuda.synchronize()      # IMPORTANT: wait for kernel to finish
    end = time.time()

    gpu_time = (end - start) * 1000   # convert to ms

    result = d_c.copy_to_host()

    # CPU reference
    cpu_start = time.time()
    expected = np.sqrt(a**2 + b**2)
    cpu_end = time.time()
    cpu_time = (cpu_end - cpu_start) * 1000

    print(f"GPU time: {gpu_time:.3f} ms")
    print(f"CPU time: {cpu_time:.3f} ms")
    print(f"Speedup: {cpu_time / gpu_time:.2f}x")
    print("Correct:", np.allclose(result, expected))

if __name__ == "__main__":
    main()
