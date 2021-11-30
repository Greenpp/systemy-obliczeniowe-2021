import time
from cython.parallel import prange
import matplotlib.pyplot as plt
import numpy as np

cdef float [:, :] KERNEL = np.array(
    [
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1],
    ], dtype=np.float32,
)


cdef float [:, :] load_img():
    return plt.imread('./data/lenna.png')

cdef float prod_sum(float [:, :] arr1, float [:, :] arr2) nogil:
    cdef int size_x = arr1.shape[0]
    cdef int size_y = arr1.shape[1]
    
    cdef int i = 0
    cdef int j = 0
    cdef float sum_ = 0.0
    for i in range(size_x):
        for j in range(size_y):
            sum_ = sum_ + arr1[i,j] * arr2[i,j]
            
    return sum_


cdef float apply_filter(float [:, :] X, float [:, :] kernel, int x_shift, int y_shift) nogil:
    cdef int x_size = kernel.shape[0]
    cdef int y_size = kernel.shape[1]
    cdef float [:, :] X_sub = X[x_shift : x_shift + x_size, y_shift : y_shift + y_size]

    return prod_sum(X_sub, kernel)

def run():
    time_start = time.perf_counter()
    cdef float [:, :] img = load_img()
    cdef float [:, :] img_pad = np.pad(img, 1)
    cdef int x_img = img.shape[0]
    cdef int y_img = img.shape[1]
    cdef list filtered = []
    cdef float res = 0
    cdef int i = 0
    cdef int j = 0
    cdef float [:, :] out = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)

    for i in prange(x_img, nogil=True):
        for j in prange(y_img):
            res = apply_filter(img_pad, KERNEL, i, j)
            out[i,j] = res
            

    filtered_arr = np.array(out)
    time_stop = time.perf_counter()
    plt.imshow(filtered_arr > 0.2, cmap='gray')
    print(f'Time: {time_stop - time_start}')
