from geda.data_providers.base import DataProvider
from typing import Literal
from PIL import Image
from geda.utils.files import read_txt_file, load_yaml


class BaseDataset:
    def __init__(
        self,
        root: str,
        split: Literal["train", "val", "test"],
        data_provider: DataProvider,
        download: bool = True,
    ):
        if download:
            data_provider.get_data()
        self.data_provider = data_provider
        self.split = split
        self.ids = data_provider.split_ids[split]
        self.root = root


class BaseSegmentationDataset(BaseDataset):
    def __init__(
        self,
        root: str,
        split: Literal["train", "val", "test"],
        data_provider: DataProvider,
        download: bool = True,
    ):
        super().__init__(root, split, data_provider, download)
        self.images_paths = data_provider.filepaths["images"][split]
        self.masks_paths = data_provider.filepaths["masks"][split]

    def get_raw_data(self, idx) -> tuple[Image.Image, Image.Image]:
        image_fpath = self.images_paths[idx]
        mask_fpath = self.masks_paths[idx]
        image = Image.open(image_fpath).convert("RGB")
        mask = Image.open(mask_fpath)
        return image, mask

    def __len__(self):
        return len(self.images_paths)


class BaseClassificationDataset(BaseDataset):
    def __init__(
        self,
        root: str,
        split: Literal["train", "val", "test"],
        data_provider: DataProvider,
        download: bool = True,
    ):
        super().__init__(root, split, data_provider, download)
        print(data_provider.filepaths)
        self.images_paths = data_provider.filepaths["images"][split]
        self.labels_paths = data_provider.filepaths["labels"][split]

    def get_raw_data(self, idx: int) -> tuple[Image.Image, int]:
        image_fpath = self.images_paths[idx]
        label_fpath = self.labels_paths[idx]
        image = Image.open(image_fpath).convert("RGB")
        label = int(read_txt_file(label_fpath)[0])
        return image, label

    def __len__(self):
        return len(self.images_paths)


class BaseKeypointsDataset(BaseDataset):
    def __init__(
        self,
        root: str,
        split: Literal["train", "val", "test"],
        data_provider: DataProvider,
        download: bool = True,
    ):
        super().__init__(root, split, data_provider, download)
        print(data_provider.filepaths)
        self.images_paths = data_provider.filepaths["images"][split]
        self.annots_paths = data_provider.filepaths["annots"][split]

    def get_raw_data(self, idx: int) -> tuple[Image.Image, dict]:
        image_fpath = self.images_paths[idx]
        label_fpath = self.annots_paths[idx]
        image = Image.open(image_fpath).convert("RGB")
        annot = load_yaml(label_fpath)
        return image, annot

    def __len__(self):
        return len(self.images_paths)
