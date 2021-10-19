# %%
import io
import time
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import requests

BASE_URL = "http://156.17.43.89:8080/sysoai/"


def download_data(url: str) -> np.ndarray:
    response = requests.get(url)
    response.raise_for_status()
    data = np.load(io.BytesIO(response.content))

    return data


def get_files_urls() -> list[str]:
    response = requests.get(BASE_URL)
    json = response.json()

    files_urls = [
        f"{BASE_URL}{file['name']}" for file in json if file['type'] == 'file'
    ]

    return files_urls


def show_image(img: np.ndarray) -> None:
    plt.imshow(img, cmap='gray')
    plt.show()


def recreate_img(data: list[np.ndarray]) -> np.ndarray:
    part_num = 1
    rows = []
    row = []
    for part in data:
        if part_num % 16:
            row.append(part)
        else:
            rows.append(np.hstack(row))
            row = []

        part_num += 1
    img = np.vstack(rows)

    return img


if __name__ == '__main__':
    time_start = time.perf_counter()
    files_urls = get_files_urls()

    with ThreadPoolExecutor() as exec:
        results: list[np.ndarray] = exec.map(download_data, files_urls)
    img = recreate_img(results)
    time_end = time.perf_counter()

    show_image(img)
    print(f'Generation time: {time_end - time_start}')

# %%
