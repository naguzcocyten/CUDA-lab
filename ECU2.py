!uv pip install -q --system numba-cuda==0.4.0
import numpy as np

import time
import os

# Enable the CUDA simulator. This MUST be set BEFORE numba imports or kernel definitions.
os.environ["NUMBA_ENABLE_CUDASIM"] = "1"
from numba import cuda
from numba import config

# ---- Configuration & Data Preparation ----

config.CUDA_ENABLE_PYNVJITLINK = 1

@cuda.jit
def whoami():
    # Compute block id in a 3D grid
    block_id = (
        cuda.blockIdx.x +
        cuda.blockIdx.y * cuda.gridDim.x +
        cuda.gridDim.x * cuda.gridDim.y
    )

    # Threads per block
    threads_per_block = (
        cuda.blockDim.x * cuda.blockDim.y
    )

    # Offset of this block
    block_offset = block_id * threads_per_block

    # Compute thread id inside block
    thread_offset = (
        cuda.threadIdx.x +
        cuda.threadIdx.y * cuda.blockDim.x +
        cuda.blockDim.x * cuda.blockDim.y
    )

    # Global thread id across all blocks
    global_id = block_offset + thread_offset


    print(f"{global_id:03d} | Block[x, y]({cuda.blockIdx.x} {cuda.blockIdx.y}) = {block_id:3d} | "
          f"Thread[x, y] ({cuda.threadIdx.x} {cuda.threadIdx.y} ) = {thread_offset:3d} BlockDim.x {cuda.blockDim.x} BlockDim.y {cuda.blockDim.y} GridDim.x {cuda.gridDim.x} GridDim.y {cuda.gridDim.y}")


b_x, b_y = 2, 2
t_x, t_y = 4, 1

blocks_per_grid = (b_x, b_y)
threads_per_block = (t_x, t_y)

total_blocks = b_x * b_y
total_threads = t_x * t_y
print(f"{total_blocks} blocks/grid")
print(f"{total_threads} threads/block")
print(f"{total_blocks * total_threads} total threads\n")

# Launch kernel
whoami[blocks_per_grid, threads_per_block]()

# Wait for GPU to finish (like cudaDeviceSynchronize)
cuda.synchronize()
