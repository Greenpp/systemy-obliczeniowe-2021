# %%
import time
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

KERNEL = np.array(
    [
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1],
    ]
)

IMG_DIR = Path('./data/scaled')


def load_imgs() -> list[np.ndarray]:
    imgs = []
    for img_path in IMG_DIR.glob('*.png'):
        img = plt.imread(img_path)
        imgs.append(img)

    return imgs


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


# %%

imgs = load_imgs()
results_mp = {}
results_lin = {}
for img in imgs:
    print(f'Processing {img.shape[0]}')
    # Multiprocessing
    time_start = time.perf_counter()
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
    results_mp[img.shape[0]] = time_stop - time_start

    # Loop
    time_start = time.perf_counter()
    img_pad = np.pad(img, 1)
    x_img, y_img = img.shape
    filtered = []
    for i in range(x_img):
        for j in range(y_img):
            res = apply_filter(img_pad, KERNEL, i, j)
            filtered.append(res)

    filtered_arr = np.array(filtered).reshape(img.shape)
    time_stop = time.perf_counter()
    results_lin[img.shape[0]] = time_stop - time_start
    # plt.imshow(filtered_arr > 0.2, cmap='gray')
    # plt.show()
    # plt.clf()

# %%
results_mp
# %%
results_lin
# %%
mp_xs = []
mp_ys = []
for k, v in sorted(results_mp.items(), key=lambda x: x[0]):
    mp_xs.append(k / 512)
    mp_ys.append(v)
# %%
# %%
lin_xs = []
lin_ys = []
for k, v in sorted(results_lin.items(), key=lambda x: x[0]):
    lin_xs.append(k / 512)
    lin_ys.append(v)
# %%
plt.plot(mp_xs, mp_ys, label='Multiprocessing')
plt.plot(lin_xs, lin_ys, label='Base')
plt.xlabel('Scale')
plt.ylabel('Time[s]')
plt.legend()
plt.show()

# %%
