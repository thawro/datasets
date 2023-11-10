from geda.data_providers.coco import COCOKeypointsDataProvider
from geda.datasets.base import BaseKeypointsDataset
from typing import Literal


class COCOKeypointsDataset(BaseKeypointsDataset):
    def __init__(self, root: str, split: Literal["train", "val", "test"]):
        data_provider = COCOKeypointsDataProvider(root)
        super().__init__(root, split, data_provider)


if __name__ == "__main__":
    from geda.utils.config import ROOT

    root = str(ROOT / "data" / "DUTS")
    train_ds = COCOKeypointsDataset(root, "train")
    val_ds = COCOKeypointsDataset(root, "val")
    test_ds = COCOKeypointsDataset(root, "test")
