from geda.utils.files import download_file, unzip, path_exists, load_yaml, save_yaml
from geda.utils.pylogger import get_pylogger
from geda.utils.images import load_img_to_array
from abc import abstractmethod
from pathlib import Path
import glob
import numpy as np
from tqdm.auto import tqdm
from functools import cached_property
from typing import Literal

log = get_pylogger(__name__)


class DataProvider:
    URL: str

    def __init__(self, urls: dict[str, str], root: str):
        self.urls = urls
        self.root = root
        if not hasattr(self, "task_root"):
            self.task_root = "Task"
        Path(self.root).mkdir(parents=True, exist_ok=True)
        self.raw_root = f"{root}/raw"

    def get_data(self, remove_zip: bool = False):
        if not self.check_if_present():
            self.zip_filepaths = self.download()  # TODO
            self.unzip(remove=remove_zip)
            log.info("Loading splits ids")
            self.split_ids = self._get_split_ids()
            log.info("Arranging files")
            self.arrange_files()
            self.create_labels()
            self.filepaths = self._get_filepaths()
        else:
            log.info("Canceling data download and files rearranging")
            self.split_ids = self._get_split_ids()
            self.filepaths = self._get_filepaths()

    def download(self):
        zip_filepaths = []
        for name, url in self.urls.items():
            ext = url.split("/")[-1].split(".")[1]
            zip_filepath = f"{self.root}/{name}.{ext}"
            if ext in ["zip", "tar", "gz"]:
                zip_filepaths.append(zip_filepath)
            if path_exists(zip_filepath):
                log.info(f"{zip_filepath} is already present. Download canceled")
            else:
                download_file(url, zip_filepath)
        return zip_filepaths

    def unzip(self, remove: bool = False):
        if path_exists(self.raw_root):
            log.info(f"{self.raw_root} is already present. Unzip canceled")
            return
        Path(self.raw_root).mkdir(parents=True, exist_ok=True)
        for zip_filepath in self.zip_filepaths:
            unzip(zip_filepath, self.root, remove)
        self.move_to_raw_root()
        if remove:
            self.zip_filepaths.clear()

    def check_if_present(self) -> bool:
        is_present = path_exists(self.task_root)
        if is_present:
            log.info(f"{self.task_root} is already present")
        return is_present

    @abstractmethod
    def move_to_raw_root(self):
        raise NotImplementedError()

    @abstractmethod
    def _get_split_ids(self):
        raise NotImplementedError()

    @property
    def splits(self) -> list[str]:
        return list(self.split_ids.keys())

    @abstractmethod
    def arrange_files(self):
        raise NotImplementedError()

    def _get_filepaths(self, dirnames: list[str] = []):
        filepaths = {}
        for dirname in dirnames:
            splits_paths = {}
            for split in self.split_ids:
                paths = glob.glob(f"{self.task_root}/{dirname}/{split}/*")
                splits_paths[split] = sorted(paths)
            filepaths[dirname] = splits_paths
        return filepaths

    @abstractmethod
    def create_labels(self):
        log.warn(f"create_labels method not implemented for {self.__class__.__name__}")


class SegmentationDataProvider(DataProvider):
    def __init__(
        self,
        urls: dict[str, str],
        root: str,
        labels_format: Literal["yolo"] | None = "yolo",
    ):
        super().__init__(urls, root)
        self.labels_format = labels_format

    def _get_filepaths(self, dirnames: list[str] = ["masks", "images", "labels"]):
        super()._get_filepaths(dirnames)

    def _get_class_counts(
        self, masks_dirname: str = "masks"
    ) -> dict[str, dict[str, int]]:
        class_counts_path = f"{self.task_root}/class_counts.yaml"
        if path_exists(class_counts_path):
            log.info(f"{class_counts_path} is present. Loading counts from that file.")
            return load_yaml(class_counts_path)
        splits_cls_freqs = {}
        for split in self.splits:
            masks_filepaths = self.filepaths[masks_dirname][split]
            classes_counts = {}
            for path in tqdm(
                masks_filepaths, desc=f"Counting class pixels for {split} split"
            ):
                mask = load_img_to_array(path)
                classes, counts = np.unique(mask, return_counts=True)
                for _class, count in zip(classes, counts):
                    if _class not in classes_counts:
                        classes_counts[_class.item()] = count.item()
                    else:
                        classes_counts[_class.item()] += count.item()
            splits_cls_freqs[split] = classes_counts
        save_yaml(splits_cls_freqs, class_counts_path)
        return splits_cls_freqs

    def _get_class_frequencies(self) -> dict[str, dict[str, float]]:
        class_frequencies_path = f"{self.task_root}/class_frequencies.yaml"
        if path_exists(class_frequencies_path):
            log.info(
                f"{class_frequencies_path} is present. Loading frequencies from that file."
            )
            return load_yaml(class_frequencies_path)
        splits_cls_freqs = {}
        for split in self.splits:
            split_cls_counts = self.class_counts[split]
            n_total_pixels = sum(split_cls_counts.values())
            cls_freqs = {k: v / n_total_pixels for k, v in split_cls_counts.items()}
            splits_cls_freqs[split] = cls_freqs
        save_yaml(splits_cls_freqs, class_frequencies_path)
        return splits_cls_freqs

    def _set_id2class(self, id2class: dict[int, str]):
        self._id2class = id2class

    @cached_property
    def class_counts(self):
        return self._get_class_counts()

    @cached_property
    def id2class(self):
        return self._id2class

    @cached_property
    def class_frequencies(self):
        return self._get_class_frequencies()

    @cached_property
    def class2id(self):
        return {v: k for k, v in self.id2class.items()}

    def save_id2class(self):
        save_yaml(self.id2class, f"{self.task_root}/id2class.yaml")

    def get_data(self):
        super().get_data()
        self.save_id2class()
        self._get_class_counts()
        self._get_class_frequencies()


class ClassificationDataProvider(DataProvider):
    def _get_filepaths(self, dirnames: list[str] = ["images", "labels"]):
        return super()._get_filepaths(dirnames)
