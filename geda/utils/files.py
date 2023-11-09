from tqdm import tqdm
import urllib.request
import tarfile
from geda.utils.pylogger import get_pylogger
import os
import shutil
from pathlib import Path
import zipfile
import yaml
import gzip

log = get_pylogger(__name__)


class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""

    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        return self.update(b * bsize - self.n)  # also sets self.n = b * bsize


def download_file(url, filepath):
    log.info(f"Downloading {url} to {filepath}")
    with TqdmUpTo(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        desc=url.split("/")[-1],
    ) as t:  # all optional kwargs
        urllib.request.urlretrieve(
            url, filename=filepath, reporthook=t.update_to, data=None
        )
        t.total = t.n
    log.info("Download finished")


def unzip_tar(file_path, dst_path, mode: str = "r"):
    with tarfile.open(file_path, mode) as tar:
        tar.extractall(dst_path)
    log.info("Unzipping finished")


def unzip_zip(file_path, dst_path):
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(dst_path)


def unzip_gz(filepath):
    with gzip.open(filepath, "rb") as f_in:
        with open(filepath.replace(".gz", ""), "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)


def unzip(file_path, dst_path, remove=False):
    log.info(f"Unzipping {file_path} to {dst_path}.")
    ext = file_path.split(".")[-1]
    if ext == "tar":
        unzip_tar(file_path, dst_path)
    elif ext == "zip":
        unzip_zip(file_path, dst_path)
    elif ext == "gz":
        unzip_gz(file_path)
    if remove:
        os.remove(file_path)
        log.info(f"Removed {file_path}")


def save_txt_to_file(txt, filename):
    with open(filename, "w") as file:
        file.write(txt)


def read_txt_file(filename) -> list[str]:
    with open(filename, "r") as file:
        lines = file.readlines()
        lines = [
            line.strip() for line in lines
        ]  # Optional: Remove leading/trailing whitespace
    return lines


def add_prefix_to_files(directory, prefix, ext=".png"):
    log.info(f"Adding {prefix} prefix to all {ext} files in {directory} directory")

    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)) and filename.endswith(ext):
            new_filename = prefix + filename
            os.rename(
                os.path.join(directory, filename), os.path.join(directory, new_filename)
            )
    log.info("Prefix addition finished")


def move(source, destination):
    shutil.move(source, destination)
    log.info(f"Moved {source} to {destination}")


def copy_directory(source_dir, destination_dir):
    shutil.copytree(source_dir, destination_dir)
    log.info(f"Copied {source_dir} to {destination_dir}")


def remove_directory(dir_path):
    shutil.rmtree(dir_path)
    log.info(f"Removed {dir_path} directory")


def path_exists(path: str):
    return os.path.exists(path)


def copy_all_files(source_dir, destination_dir, ext=".png"):
    filenames = os.listdir(source_dir)
    for filename in tqdm(filenames, desc="Copying files"):
        if filename.lower().endswith(ext):
            source = source_dir / filename
            destination = destination_dir / filename
            shutil.copy2(source, destination)
    log.info(
        f"Copied all {ext} files ({len(filenames)}) from {source_dir} to {destination_dir}"
    )


def copy_files(source_filepaths, dest_filepaths):
    for source_filepath, dest_filepath in tqdm(
        zip(source_filepaths, dest_filepaths), desc="Copying files"
    ):
        try:
            shutil.copy2(source_filepath, dest_filepath)
        except FileNotFoundError as e:
            log.warn(f"{source_filepath} not found")
    log.info(f"Copied files ({len(source_filepaths)})")


def create_dir(path: Path, return_str: bool = True):
    path.mkdir(parents=True, exist_ok=True)
    if return_str:
        return str(path)
    return path


def load_yaml(path: Path | str) -> dict:
    with open(path, "r") as file:
        yaml_dct = yaml.safe_load(file)
    return yaml_dct


def save_yaml(dct: dict, path: Path | str):
    with open(path, "w") as file:
        yaml.dump(dct, file)
