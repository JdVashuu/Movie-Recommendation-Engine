import os
import urllib.request
import zipfile

DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
DATA_DIR = "data/raw"
ZIP_PATH = os.path.join(DATA_DIR, "ml-100k.zip")


def download_movielens():
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(ZIP_PATH):
        print("Downloading Movielens dataset...")
        urllib.request.urlretrieve(DATA_URL, ZIP_PATH)

    extracted_dir = os.path.join(DATA_DIR, "ml-100k")
    if not os.path.exists(extracted_dir):
        print("Extracting dataset...")
        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(DATA_DIR)

    print("Data Ready")


if __name__ == "__main__":
    download_movielens()
