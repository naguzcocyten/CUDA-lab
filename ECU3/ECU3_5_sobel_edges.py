import numpy as np
import numba.cuda as cuda
import time
import urllib.request
from PIL import Image
from matplotlib import pyplot as plt
import cv2

import numpy as np
import numba.cuda as cuda
import time
import urllib.request
from PIL import Image
import cv2
from matplotlib import pyplot as plt


@numba.cuda.jit
def sobel_kernel(img, out):
    """Apply Sobel edge detection - each thread processes one pixel"""
    row, col = cuda.grid(2)
    H, W = img.shape

    if 0 < row < H-1 and 0 < col < W-1:

        # Horizontal gradient (Gx)
        gx = ( -img[row-1, col-1] + img[row-1, col+1]
               -2*img[row, col-1] + 2*img[row, col+1]
               -img[row+1, col-1] + img[row+1, col+1] )

        # Vertical gradient (Gy)
        gy = ( -img[row-1, col-1] - 2*img[row-1, col] - img[row-1, col+1]
               + img[row+1, col-1] + 2*img[row+1, col] + img[row+1, col+1] )

        # Edge magnitude
        out[row, col] = (gx*gx + gy*gy)**0.5


def sobel_opencv(img):
    """OpenCV CPU version using Sobel"""
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx**2 + gy**2)


# Load 4K image from internet
urllib.request.urlretrieve("https://picsum.photos/3840/2160", "image.jpg")
img = Image.open("image.jpg").convert('L')   # Convert to grayscale
img = np.array(img, dtype=np.float32)

H, W = img.shape
print(f"Image: {W}×{H} ({W*H:,} pixels)")


d_img = cuda.to_device(img)
d_out = cuda.to_device(np.zeros_like(img))

threads = (32, 32)
blocks = ((W + 15) // 16, (H + 15) // 16)

print(f"Grid: {blocks} blocks × {threads} threads")


# Warmup
sobel_kernel[blocks, threads](d_img, d_out)
cuda.synchronize()


# Timed run (GPU)
start = time.time()
sobel_kernel[blocks, threads](d_img, d_out)
cuda.synchronize()
gpu_time = (time.time() - start) * 1000

out_gpu = d_out.copy_to_host()


# CPU Sobel timing
start = time.time()
out_cpu = sobel_opencv(img)
cpu_time = (time.time() - start) * 1000


# Results
print("\n" + "="*60)
print("Results")
print("="*60)
print(f"GPU: {gpu_time:.2f} ms")
print(f"CPU: {cpu_time:.2f} ms")
print(f"Speedup: {cpu_time/gpu_time:.1f}x")
print(f"Correct: {np.allclose(out_gpu, out_cpu, atol=1e-3)}")


# Resize for display
H, W = img.shape
target_w = 256
target_h = int(target_w * H / W)


def resize_for_plot(array):
    normalized = (array / array.max() * 255).astype(np.uint8)
    return np.array(Image.fromarray(normalized).resize((target_w, target_h), Image.LANCZOS))


plt.figure(figsize=(20, 10))

plt.subplot(1, 3, 1)
plt.imshow(resize_for_plot(img), cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(resize_for_plot(out_gpu), cmap='gray')
plt.title('GPU Sobel Edges')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(resize_for_plot(out_cpu), cmap='gray')
plt.title('OpenCV CPU Sobel Edges')
plt.axis('off')

plt.tight_layout()
plt.show()
