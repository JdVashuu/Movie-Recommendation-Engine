import os
import urllib.request
import zipfile

from core.config import DATA_DIR, DATA_URL, ML_100K_DIR, ZIP_PATH


def download_movielens():
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(ZIP_PATH):
        print("Downloading Movielens dataset...")
        urllib.request.urlretrieve(DATA_URL, ZIP_PATH)

    if not os.path.exists(ML_100K_DIR):
        print("Extracting dataset...")
        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(DATA_DIR)

    print("Data Ready")


if __name__ == "__main__":
    download_movielens()
