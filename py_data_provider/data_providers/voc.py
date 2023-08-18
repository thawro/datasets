from py_data_provider.data_providers.base import TarDataProvider
from py_data_provider.utils.files import (
    path_exists,
    move,
    remove_directory,
    read_text_file,
    copy_files,
    create_dir,
)
from py_data_provider.parsers.yolo import parse_segmentation_mask_to_yolo_format
from typing import Literal
from pathlib import Path
import glob
from py_data_provider.utils.pylogger import get_pylogger
from PIL import Image
import numpy as np
from tqdm.auto import tqdm


log = get_pylogger(__name__)

VOC_URLS = {
    2007: {
        "trainval_2007": "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar",
        "test_2007": "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar",
    },
    2012: {
        "trainval_2012": "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
    },
}


class VOCDataProvider(TarDataProvider):
    def __init__(
        self,
        root: str,
        year: Literal[2007, 2012],
        task: Literal["Action", "Layout", "Main", "Segmentation"],
    ):
        self.year = year
        self.task = task
        self.task_root = f"{root}/{task}"
        super().__init__(urls=VOC_URLS[year], root=root)

    def check_if_present(self) -> bool:
        return path_exists(self.task_root)

    def move_to_raw_root(self):
        src_dir = f"{self.root}/VOCdevkit/VOC{self.year}/"
        move(src_dir, self.raw_root)
        remove_directory(f"{self.root}/VOCdevkit")

    def set_split_ids(self):
        task_dir = f"{self.raw_root}/ImageSets/{self.task}"
        train_ids = read_text_file(f"{task_dir}/train.txt")
        val_ids = read_text_file(f"{task_dir}/val.txt")
        if self.task == "Layout":
            train_ids = [_id.split(" ")[0] for _id in train_ids]
            val_ids = [_id.split(" ")[0] for _id in val_ids]
            train_ids = sorted(list(set(train_ids)))
            val_ids = sorted(list(set(val_ids)))
        self.split_ids = {"train": train_ids, "val": val_ids}

    def arrange_files(self):
        for split, ids in self.split_ids.items():
            names = ["annots", "images", "labels"]
            dst_paths = {name: create_dir(Path(self.task_root) / name / split) for name in names}
            src_paths = {
                "annots": f"{self.raw_root}/Annotations",
                "images": f"{self.raw_root}/JPEGImages",
            }

            src_annots_filepaths = [f"{src_paths['annots']}/{_id}.xml" for _id in ids]
            src_images_filepaths = [f"{src_paths['images']}/{_id}.jpg" for _id in ids]

            dst_annots_filepaths = [f"{dst_paths['annots']}/{_id}.xml" for _id in ids]
            dst_images_filepaths = [f"{dst_paths['images']}/{_id}.jpg" for _id in ids]

            log.info(f"Moving {split} annots from {src_paths['annots']} to {dst_paths['annots']}")
            log.info(f"Moving {split} images from {src_paths['images']} to {dst_paths['images']}")
            copy_files(src_annots_filepaths, dst_annots_filepaths)
            copy_files(src_images_filepaths, dst_images_filepaths)

    def set_filepaths(self, dirnames: list[str] = ["annots", "images", "labels"]):
        filepaths = {}
        for dirname in dirnames:
            splits_paths = {}
            for split in self.split_ids:
                paths = glob.glob(f"{self.task_root}/{dirname}/{split}/*")
                splits_paths[split] = sorted(paths)
            filepaths[dirname] = splits_paths
        self.filepaths = filepaths


_LABELS = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

_SEG_ID2LABEL = {i: label for i, label in enumerate(_LABELS, start=1)} | {
    0: "background",
    255: "void",
}


class VOCSegmentationDataProvider(VOCDataProvider):
    def __init__(
        self,
        root: str,
        year: Literal[2007, 2012] = 2012,
        mode: Literal["semantic", "instance"] = "semantic",
        labels_format: Literal["yolo"] = "yolo",
    ):
        self.mode = mode
        self.id2label = _SEG_ID2LABEL
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.labels_format = labels_format
        super().__init__(root=root, year=year, task="Segmentation")

    def arrange_files(self):
        super().arrange_files()
        mask_name = "SegmentationClass" if self.mode == "semantic" else "SegmentationObject"

        for split, ids in self.split_ids.items():
            src_masks_path = f"{self.raw_root}/{mask_name}"
            dst_masks_path = create_dir(Path(self.task_root) / "masks" / split)

            src_filepaths = [f"{src_masks_path}/{_id}.png" for _id in ids]
            dst_filepaths = [f"{dst_masks_path}/{_id}.png" for _id in ids]

            log.info(f"Moving {split} masks from {src_masks_path} to {dst_masks_path}")
            copy_files(src_filepaths, dst_filepaths)

    def set_filepaths(self):
        super().set_filepaths(["annots", "images", "labels", "masks"])

    def create_labels(self):
        masks_filepaths = sorted(glob.glob(f"{self.task_root}/masks/*/*"))
        for filename in tqdm(masks_filepaths, desc="Creating labels"):
            mask = np.array(Image.open(filename))
            label_filepath = filename.replace("masks", "labels").replace(".png", ".txt")
            if self.labels_format == "yolo":
                annot_txt = parse_segmentation_mask_to_yolo_format(
                    mask=mask,
                    background=self.label2id["background"],
                    contour=self.label2id["void"],
                )
                with open(label_filepath, "w") as file:
                    file.write(annot_txt)
