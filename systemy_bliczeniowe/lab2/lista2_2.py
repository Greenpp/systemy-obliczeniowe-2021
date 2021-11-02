# %%
import time
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np

KERNEL = np.array(
    [
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1],
    ]
)


def load_img() -> np.ndarray:
    img = plt.imread('./data/lenna.png')

    return img


def apply_filter(
    X: np.ndarray, kernel: np.ndarray, x_shift: int, y_shift: int
) -> float:
    x_size, y_size = kernel.shape
    X_sub = X[x_shift : x_shift + x_size, y_shift : y_shift + y_size]

    res = X_sub * kernel

    return res.sum()


def apply_wrapper(arg: tuple) -> float:
    X, kernel, x_shift, y_shift = arg

    return apply_filter(X, kernel, x_shift, y_shift)


time_start = time.perf_counter()
img = load_img()
img_pad = np.pad(img, 1)
x_img, y_img = img.shape
filtered = []
args = []
for i in range(x_img):
    for j in range(y_img):
        args.append((img_pad, KERNEL, i, j))

with Pool() as p:
    filtered = p.map(apply_wrapper, args)

filtered_arr = np.array(filtered).reshape(img.shape)
time_stop = time.perf_counter()
plt.imshow(filtered_arr > 0.2, cmap='gray')
print(f'Time: {time_stop - time_start}')

# %%
