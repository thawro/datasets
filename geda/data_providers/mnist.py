from geda.data_providers.base import ClassificationDataProvider
from geda.utils.files import move, copy_files, create_dir, save_txt_to_file
from geda.utils.pylogger import get_pylogger
from tqdm.auto import tqdm
import numpy as np
from pathlib import Path
from PIL import Image

log = get_pylogger(__name__)


_URLS = {
    "train_images": "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
}

SPLIT_NUM_SAMPLES = {"train": 60_000, "test": 10_000}


SPLITS = ["train", "test"]


class MNISTDataProvider(ClassificationDataProvider):
    URL = "http://yann.lecun.com/exdb/mnist/"

    def __init__(self, root: str):
        self.task = "Classification"
        self.task_root = f"{root}/{self.task}"
        super().__init__(urls=_URLS, root=root)

    def move_to_raw_root(self):
        filepaths = list(_URLS.keys())
        for filepath in filepaths:
            src_path = f"{self.root}/{filepath}"
            dst_path = f"{self.raw_root}/{filepath}"
            move(src_path, dst_path)

    def _get_split_ids(self):
        split_ids = {}
        for split in SPLITS:
            ids = [str(i) for i in range(SPLIT_NUM_SAMPLES[split])]
            split_ids[split] = ids
        return split_ids

    def arrange_files(self):
        def load_ubyte_to_array(filepath, labels=False):
            with open(filepath, "rb") as file:
                # file = open(filepath, "r")
                if labels:
                    file.read(8)
                else:
                    file.read(16)
                buf = file.read()
                data = np.frombuffer(buf, dtype=np.uint8)
                if not labels:
                    data = data.reshape(-1, 28, 28, 1)
                return data

        for split, ids in self.split_ids.items():
            names = ["images", "labels"]
            dst_paths = {name: create_dir(Path(self.task_root) / name / split) for name in names}

            images_ubyte_filepath = f"{self.raw_root}/{split}_images"
            labels_ubyte_filepath = f"{self.raw_root}/{split}_labels"

            images = load_ubyte_to_array(images_ubyte_filepath, labels=False)
            labels = load_ubyte_to_array(labels_ubyte_filepath, labels=True)
            for _id, image, label in tqdm(
                zip(ids, images, labels),
                desc=f"Saving images and labels for {split} split",
                total=len(ids),
            ):
                h, w, c = image.shape
                if c == 1:
                    image = image.squeeze(2)
                dst_label_filepath = f"{dst_paths['labels']}/{_id}.txt"
                dst_image_filepath = f"{dst_paths['images']}/{_id}.jpg"

                Image.fromarray(image).save(dst_image_filepath)
                save_txt_to_file(str(label), dst_label_filepath)


if __name__ == "__main__":
    from geda.utils.config import ROOT

    root = str(ROOT / "data" / "MNIST")
    dp = MNISTDataProvider(root)
    dp.get_data()
