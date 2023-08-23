from geda.data_providers.nyud import NYUDv2DataProvider
from geda.datasets.base import BaseSegmentationDataset
from typing import Literal


class NYUDv2Dataset(BaseSegmentationDataset):
    def __init__(
        self,
        root: str,
        split: Literal["train", "test"],
        labels_format: Literal["yolo"] = "yolo",
    ):
        data_provider = NYUDv2DataProvider(root, labels_format=labels_format)
        super().__init__(root, split, data_provider)


if __name__ == "__main__":
    from geda.utils.config import ROOT

    root = str(ROOT / "data" / "DUTS")
    ds = NYUDv2Dataset(root, "train")
    ds = NYUDv2Dataset(root, "test")
