from geda.data_providers.mpii import MPIIDataProvider
from geda.datasets.base import BaseKeypointsDataset
from typing import Literal


class MPIIDataset(BaseKeypointsDataset):
    def __init__(self, root: str, split: Literal["train", "val", "test"]):
        data_provider = MPIIDataProvider(root)
        super().__init__(root, split, data_provider)


if __name__ == "__main__":
    from geda.utils.config import ROOT

    root = str(ROOT / "data" / "DUTS")
    train_ds = MPIIDataset(root, "train")
    val_ds = MPIIDataset(root, "val")
    test_ds = MPIIDataset(root, "test")
