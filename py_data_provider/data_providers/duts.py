from py_data_provider.data_providers.base import DataProvider
from py_data_provider.utils.files import (
    path_exists,
    move,
    copy_files,
    create_dir,
)
from py_data_provider.parsers.yolo import parse_segmentation_masks_to_yolo
from pathlib import Path
import glob
from py_data_provider.utils.pylogger import get_pylogger
from PIL import Image
import numpy as np
from tqdm.auto import tqdm
import os
from typing import Literal


log = get_pylogger(__name__)


_URLS = {
    "train": "http://saliencydetection.net/duts/download/DUTS-TR.zip",
    "test": "http://saliencydetection.net/duts/download/DUTS-TE.zip",
}

split2dirname = {"train": "DUTS-TR", "test": "DUTS-TE"}


class DUTSDataProvider(DataProvider):
    def __init__(self, root: str, binarize: bool = True, labels_format: Literal["yolo"] = "yolo"):
        self.splits = ["train", "test"]
        self.task = "Segmentation"
        self.task_root = f"{root}/{self.task}"
        self.binarize = binarize
        self.labels_format = labels_format
        super().__init__(urls=_URLS, root=root)

    def check_if_present(self) -> bool:
        return path_exists(self.task_root)

    def move_to_raw_root(self):
        for split in self.splits:
            src_dir = f"{self.root}/{split2dirname[split]}/"
            move(src_dir, self.raw_root)

    def _get_split_ids(self):
        split_ids = {}
        for split in self.splits:
            split_name = split2dirname[split]
            filepaths = sorted(glob.glob(f"{self.raw_root}/{split_name}/{split_name}-Mask/*png"))
            ids = [os.path.basename(path).split(".")[0] for path in filepaths]
            split_ids[split] = ids
        return split_ids

    def arrange_files(self):
        for split, ids in self.split_ids.items():
            names = ["masks", "images", "labels"]
            dst_paths = {name: create_dir(Path(self.task_root) / name / split) for name in names}
            split_name = split2dirname[split]
            src_paths = {
                "masks": f"{self.raw_root}/{split_name}/{split_name}-Mask",
                "images": f"{self.raw_root}/{split_name}/{split_name}-Image",
            }

            src_masks_filepaths = [f"{src_paths['masks']}/{_id}.png" for _id in ids]

            src_images_filepaths = [f"{src_paths['images']}/{_id}.jpg" for _id in ids]

            dst_masks_filepaths = [f"{dst_paths['masks']}/{_id}.png" for _id in ids]
            dst_images_filepaths = [f"{dst_paths['images']}/{_id}.jpg" for _id in ids]

            log.info(f"Moving {split} masks from {src_paths['masks']} to {dst_paths['masks']}")
            log.info(f"Moving {split} images from {src_paths['images']} to {dst_paths['images']}")
            copy_files(src_masks_filepaths, dst_masks_filepaths)
            copy_files(src_images_filepaths, dst_images_filepaths)

            if self.binarize:
                log.info(f"Applying binarization (threshold = 128) for {split} masks")
                for mask_filepath in tqdm(dst_masks_filepaths, desc="Binarization"):
                    mask = np.array(Image.open(mask_filepath).convert("L"))
                    binary_mask = ((mask > 128) * 255).astype(np.uint8)
                    Image.fromarray(binary_mask).save(mask_filepath)

    def _get_filepaths(self, dirnames: list[str] = ["masks", "images", "labels"]):
        filepaths = {}
        for dirname in dirnames:
            splits_paths = {}
            for split in self.split_ids:
                paths = glob.glob(f"{self.task_root}/{dirname}/{split}/*")
                splits_paths[split] = sorted(paths)
            filepaths[dirname] = splits_paths
        return filepaths

    def create_labels(self):
        masks_filepaths = sorted(glob.glob(f"{self.task_root}/masks/*/*"))
        if self.labels_format == "yolo":
            parse_segmentation_masks_to_yolo(masks_filepaths, background=0, contour=None)
        else:
            log.warn(f"Only YOLO label format is implemented ({self.labels_format} passed)")


if __name__ == "__main__":
    from py_data_provider.utils.config import ROOT

    root = str(ROOT / "data" / "DUTS")
    dp = DUTSDataProvider(root)
