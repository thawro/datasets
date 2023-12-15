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
from joblib import Parallel, delayed

log = get_pylogger(__name__)

_filepaths = list[str]


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


def download_file(url: str, filepath: str):
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


def unzip_tar(filepath: str, dst_path, mode: str = "r"):
    with tarfile.open(filepath, mode) as tar:
        tar.extractall(dst_path)
    log.info("Unzipping finished")


def unzip_zip(filepath: str, dst_path: str):
    with zipfile.ZipFile(filepath, "r") as zip_ref:
        zip_ref.extractall(dst_path)


def unzip_gz(filepath: str):
    with gzip.open(filepath, "rb") as f_in:
        with open(filepath.replace(".gz", ""), "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)


def unzip(filepath: str, dst_path: str, remove: bool = False):
    log.info(f"Unzipping {filepath} to {dst_path}.")
    ext = filepath.split(".")[-1]
    if ext == "tar":
        unzip_tar(filepath, dst_path)
    elif ext == "zip":
        unzip_zip(filepath, dst_path)
    elif ext == "gz":
        unzip_gz(filepath)
    if remove:
        os.remove(filepath)
        log.info(f"Removed {filepath}")


def unzip_many(filepaths: list[str], dst_paths: list[str] | str, remove: bool = False):
    desc = "Unzipping files"
    total = len(filepaths)
    if isinstance(dst_paths, str):
        dst_paths = [dst_paths] * total
    Parallel(n_jobs=-1)(
        delayed(unzip)(zip_filepath, dst_path, remove)
        for zip_filepath, dst_path in tqdm(
            zip(filepaths, dst_paths), desc=desc, total=total
        )
    )


def save_txt_to_file(txt: str, filename: str):
    with open(filename, "w") as file:
        file.write(txt)


def read_txt_file(filepath: str) -> list[str]:
    with open(filepath, "r") as file:
        lines = file.readlines()
        lines = [
            line.strip() for line in lines
        ]  # Optional: Remove leading/trailing whitespace
    return lines


def add_prefix_to_files(directory: str, prefix: str, ext: str = ".png"):
    log.info(f"Adding {prefix} prefix to all {ext} files in {directory} directory")

    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)) and filename.endswith(ext):
            new_filename = prefix + filename
            os.rename(
                os.path.join(directory, filename), os.path.join(directory, new_filename)
            )
    log.info("Prefix addition finished")


def move(src: str, dst: str):
    shutil.move(src, dst)
    log.info(f"Moved {src} to {dst}")


def move_many(sources: _filepaths, destinations: _filepaths):
    total = len(sources)
    desc = "Moving files"
    log.info(f"Moving {total} files/directories")
    Parallel(n_jobs=-1)(
        delayed(shutil.move)(src, dst)
        for src, dst in tqdm(zip(sources, destinations), desc=desc, total=total)
    )
    log.info(f"Moved {total} files")


def copy_directory(src_dir: str, dst_dir: str):
    shutil.copytree(src_dir, dst_dir)
    log.info(f"Copied {src_dir} to {dst_dir}")


def remove_directory(dirpath: str):
    shutil.rmtree(dirpath)
    log.info(f"Removed {dirpath} directory")


def path_exists(path: str):
    return os.path.exists(path)


def copy_file(src: str, dst: str):
    try:
        shutil.copy2(src, dst)
    except FileNotFoundError as e:
        log.warn(f"{src} not found")


def copy_all_files(src_dir: Path, dst_dir: Path, ext: str = ".png"):
    def _copy(filename: str):
        if filename.lower().endswith(ext):
            source = str(src_dir / filename)
            destination = str(dst_dir / filename)
            copy_file(source, destination)

    desc = "Copying files"
    filenames = os.listdir(src_dir)
    Parallel(n_jobs=-1)(delayed(_copy)(fname) for fname in tqdm(filenames, desc=desc))
    log.info(f"Copied all {ext} files ({len(filenames)}) from {src_dir} to {dst_dir}")


def copy_files(src_filepaths: _filepaths, dst_filepaths: _filepaths):
    desc = "Copying files"
    total = len(src_filepaths)
    Parallel(n_jobs=-1)(
        delayed(copy_file)(src, dst)
        for src, dst in tqdm(zip(src_filepaths, dst_filepaths), desc=desc, total=total)
    )
    log.info(f"Copied {total} files")


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


def save_yamls(dcts: list[dict], paths: list[str]):
    desc = "Copying yaml files"
    total = len(dcts)
    Parallel(n_jobs=-1)(
        delayed(save_yaml)(dct, path)
        for dct, path in tqdm(zip(dcts, paths), desc=desc, total=total)
    )
