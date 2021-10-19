# %%
import io
import time
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import requests

time_start = time.perf_counter()
base_url = "http://156.17.43.89:8080/sysoai/"

response = requests.get(base_url)
json = response.json()

rows = []
part_num = 1
row = []
for file in json:
    if file['type'] == 'file':
        url = '%s%s' % (base_url, file['name'])
        response = requests.get(url)
        response.raise_for_status()
        data = np.load(io.BytesIO(response.content))
        if part_num % 16:
            row.append(data)
        else:
            rows.append(np.hstack(row))
            row = []

        part_num += 1
all_data = np.vstack(rows)
time_end = time.perf_counter()

plt.imshow(all_data, cmap='gray')
plt.show()

print(f'Generation time: {time_end - time_start}')

# %%
