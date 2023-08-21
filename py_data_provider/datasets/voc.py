from py_data_provider.data_providers.voc import (
    VOCInstanceSegmentationDataProvider,
    VOCPersonPartDataProvider,
    VOCSemanticSegmentationDataProvider,
)
from py_data_provider.datasets.base import BaseSegmentationDataset
from typing import Literal


class VOCInstanceSegmentationDataset(BaseSegmentationDataset):
    def __init__(
        self, root: str, split: Literal["train", "val"], labels_format: Literal["yolo"] = "yolo"
    ):
        super().__init__(root, split, VOCInstanceSegmentationDataProvider(root, labels_format))


class VOCSemanticSegmentationDataset(BaseSegmentationDataset):
    def __init__(
        self, root: str, split: Literal["train", "val"], labels_format: Literal["yolo"] = "yolo"
    ):
        super().__init__(root, split, VOCSemanticSegmentationDataProvider(root, labels_format))


class VOCPersonPartSegmentationDataset(BaseSegmentationDataset):
    def __init__(
        self, root: str, split: Literal["train", "val"], labels_format: Literal["yolo"] = "yolo"
    ):
        super().__init__(root, split, VOCPersonPartDataProvider(root, labels_format))


if __name__ == "__main__":
    from py_data_provider.utils.config import ROOT

    root = str(ROOT / "data" / "voc_2012")
    ds = VOCInstanceSegmentationDataset(root, "train")
