from geda.data_providers.base import SegmentationDataProvider
from geda.utils.files import (
    move,
    copy_files,
    create_dir,
)
from geda.utils.images import save_gray_array_as_color_png
from geda.parsers.yolo import parse_segmentation_masks_to_yolo
from pathlib import Path
import glob
from geda.utils.pylogger import get_pylogger
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

_ID2CLASS = {0: "background", 1: "salient"}
_PALETTE = [(0, 0, 0), (255, 255, 255)]  # black for background, white for salient

SPLITS = ["train", "test"]


class DUTSDataProvider(SegmentationDataProvider):
    URL = "http://saliencydetection.net/duts/"

    def __init__(
        self,
        root: str,
        binarize: bool = True,
        labels_format: Literal["yolo"] | None = None,
    ):
        self.task = "Segmentation"
        self.task_root = f"{root}/{self.task}"
        self.binarize = binarize
        super().__init__(urls=_URLS, root=root, labels_format=labels_format)

    def move_to_raw_root(self):
        for split in SPLITS:
            src_dir = f"{self.root}/{split2dirname[split]}/"
            move(src_dir, self.raw_root)

    def _get_split_ids(self):
        split_ids = {}
        for split in SPLITS:
            split_name = split2dirname[split]
            filepaths = sorted(glob.glob(f"{self.raw_root}/{split_name}/{split_name}-Mask/*png"))
            ids = [os.path.basename(path).split(".")[0] for path in filepaths]
            split_ids[split] = ids
        return split_ids

    def arrange_files(self):
        for split, ids in self.split_ids.items():
            names = ["masks", "images", "labels"]
            dst_paths = {name: create_dir(Path(self.task_root) / name / split) for name in names}
            self._set_id2class(id2class=_ID2CLASS)
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
                    binary_mask = ((mask > 128) * 1).astype(np.uint8)
                    save_gray_array_as_color_png(
                        binary_mask, palette=_PALETTE, filename=mask_filepath
                    )

    def create_labels(self):
        masks_filepaths = sorted(glob.glob(f"{self.task_root}/masks/*/*"))
        if self.labels_format == "yolo":
            parse_segmentation_masks_to_yolo(masks_filepaths, background=0, contour=None)
        else:
            log.warn(f"Only YOLO label format is implemented ({self.labels_format} passed)")


if __name__ == "__main__":
    from geda.utils.config import ROOT

    root = str(ROOT / "data" / "DUTS")
    dp = DUTSDataProvider(root)
