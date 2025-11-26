!uv pip install -q --system numba-cuda==0.4.0
import numpy as np

import time
import os
import numpy as np

# Enable the CUDA simulator. This MUST be set BEFORE numba imports or kernel definitions.
os.environ["NUMBA_ENABLE_CUDASIM"] = "1"
from numba import cuda
from numba import config

# --- Configuration & Data Preparation ---

config.CUDA_ENABLE_PYNVJITLINK = 1

config.CUDA_ENABLE_PYNVJITLINK = 1  # opcional según tu entorno

# Kernel CUDA: suma elemento a elemento
@cuda.jit
def add_arrays(a, b, c):
    i = cuda.grid(1)  # índice global 1D
    if i < a.size:
        c[i] = a[i] + b[i]

def main():
    n = 1_000_0
    # Memoria en CPU
    a = np.ones(n, dtype=np.float32)
    b = 2 * np.ones(n, dtype=np.float32)
    c = np.zeros(n, dtype=np.float32)

    # Reservar memoria en GPU
    d_a = cuda.to_device(a)
    d_b = cuda.to_device(b)
    d_c = cuda.device_array_like(c)

    # Configuración del grid
    threads_per_block = 256
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block

    # Medir tiempo de ejecución en GPU
    start = time.time()
    add_arrays[blocks_per_grid, threads_per_block](d_a, d_b, d_c)
    cuda.synchronize()
    gpu_time = time.time() - start

    # Copiar resultado a CPU
    d_c.copy_to_host(c)

    print(f"Resultado c[0:5]: {c[:5]}")
    print(f"Tiempo en GPU: {gpu_time*1000:.3f} ms")

if __name__ == "__main__":
    main()



