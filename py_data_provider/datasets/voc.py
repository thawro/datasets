from py_data_provider.data_providers.voc import VOCSegmentationDataProvider
from typing import Literal
import numpy as np
from PIL import Image


class VOCSegmentationDataset:
    def __init__(
        self,
        root: str,
        split: Literal["train", "val"],
        year: Literal[2007, 2012] = 2012,
        mode: Literal["semantic", "instance"] = "semantic",
        labels_format: Literal["yolo"] = "yolo",
    ):
        self.split = split
        data_provider = VOCSegmentationDataProvider(root, year, mode, labels_format)
        self.ids = data_provider.split_ids[split]
        self.images_paths = data_provider.filepaths["images"][split]
        self.masks_paths = data_provider.filepaths["masks"][split]
        self.root = root
        self.dp = data_provider

    def __getitem__(self, idx):
        image_fpath = self.images_paths[idx]
        mask_fpath = self.masks_paths[idx]
        image = Image.open(image_fpath).convert("RGB")
        mask = Image.open(mask_fpath)
        return image, mask

    def __len__(self):
        return len(self.images_paths)
