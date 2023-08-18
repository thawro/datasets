from py_data_provider.data_providers.voc import VOCSegmentationDataProvider
from py_data_provider.datasets.base import BaseSegmentationDataset
from typing import Literal


class VOCSegmentationDataset(BaseSegmentationDataset):
    def __init__(
        self,
        root: str,
        split: Literal["train", "val"],
        year: Literal[2007, 2012] = 2012,
        mode: Literal["semantic", "instance"] = "semantic",
        labels_format: Literal["yolo"] = "yolo",
    ):
        data_provider = VOCSegmentationDataProvider(root, year, mode, labels_format)
        super().__init__(root, split, data_provider)


if __name__ == "__main__":
    from py_data_provider.utils.config import ROOT

    root = str(ROOT / "data" / "voc_2012")
    ds = VOCSegmentationDataset(root, "train", 2012)
    ds = VOCSegmentationDataset(root, "val", 2012)
