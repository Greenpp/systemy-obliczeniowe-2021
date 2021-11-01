# %%
import numpy as np


def apply_filter(
    X: np.ndarray,
    kernel: np.ndarray,
    x_offset: int,
    y_offset: int,
) -> np.ndarray:
    # NOTE no padding
    x_size, y_size = kernel.shape
    X_sub = X[x_offset : x_offset + x_size, y_offset : y_offset + y_size]
    prod = X_sub * kernel
    res = prod.sum()

    return res



# %%
x = np.ones((5,5))
# %%
x
# %%
k = np.ones((3,3))
# %%
k
# %%
apply_filter(x, k, 3, 0)
# %%
