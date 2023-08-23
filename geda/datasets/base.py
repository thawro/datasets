from geda.data_providers.base import DataProvider
from typing import Literal
from PIL import Image


class BaseSegmentationDataset:
    def __init__(
        self,
        root: str,
        split: Literal["train", "val", "test"],
        data_provider: DataProvider,
    ):
        self.split = split
        self.ids = data_provider.split_ids[split]
        self.images_paths = data_provider.filepaths["images"][split]
        self.masks_paths = data_provider.filepaths["masks"][split]
        self.root = root

    def __getitem__(self, idx):
        image_fpath = self.images_paths[idx]
        mask_fpath = self.masks_paths[idx]
        image = Image.open(image_fpath).convert("RGB")
        mask = Image.open(mask_fpath)
        return image, mask

    def __len__(self):
        return len(self.images_paths)
