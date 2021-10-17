import itertools
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from tqdm import tqdm


def download_file(url: str, output_path: Path) -> None:
    res = requests.get(url, allow_redirects=True)
    if res.ok:
        file_name = url.split('/')[-1]
        file_path = output_path / file_name
        with open(file_path, 'wb') as f:
            f.write(res.content)


def download_files(urls: list[str], output_dir: str) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    with ThreadPoolExecutor() as exec:
        list(
            tqdm(
                exec.map(download_file, urls, itertools.repeat(output_path)),
                total=len(urls),
            )
        )
