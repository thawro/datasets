from geda.data_providers.base import DataProvider

from geda.utils.files import (
    move,
    remove_directory,
    read_text_file,
    copy_files,
    create_dir,
    unzip,
    path_exists,
)
from geda.parsers.yolo import parse_segmentation_masks_to_yolo
from typing import Literal
from pathlib import Path
import glob
from geda.utils.pylogger import get_pylogger


log = get_pylogger(__name__)

_URLS = {
    2007: {
        "trainval_2007": "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar",
        "test_2007": "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar",
    },
    2012: {
        "trainval_2012": "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
    },
}

_PERSON_PART_URL = {"person_part": "http://www.liangchiehchen.com/data/pascal_person_part.zip"}

_SEG_SEMANTIC_LABELS = [
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
_SEG_PERSON_PART_LABELS = ["Head", "Torso", "Upper Arms", "Lower Arms", "Upper Legs", "Lower Legs"]

_SEG_SEMANTIC_ID2LABEL = {i: label for i, label in enumerate(_SEG_SEMANTIC_LABELS, start=1)} | {
    0: "background",
    255: "void",
}

_SEG_PERSON_PART_ID2LABEL = {
    i: label for i, label in enumerate(_SEG_PERSON_PART_LABELS, start=1)
} | {0: "background", None: "void"}

_SEG_INSTANCE_ID2LABEL = {i: str(i) for i in range(1, 100)} | {0: "background", 255: "void"}

_SEG_ID2LABEL = {
    "semantic": _SEG_SEMANTIC_ID2LABEL,
    "person_part": _SEG_PERSON_PART_ID2LABEL,
    "instance": _SEG_INSTANCE_ID2LABEL,
}


class VOCDataProvider(DataProvider):
    URL = "http://host.robots.ox.ac.uk/pascal/VOC/"

    def __init__(
        self,
        root: str,
        task: Literal[
            "Action",
            "Layout",
            "Main",
            "SegmentationClass",
            "SegmentationObject",
            "SegmentationPersonPart",
        ],
        labels_format: Literal["yolo"] = "yolo",
    ):
        self.year = 2012
        self.task_root = f"{root}/{task}"
        self.labels_format = labels_format
        if task in ["SegmentationClass", "SegmentationObject"]:
            self.task = "Segmentation"
        else:
            self.task = task
        urls = _URLS[2012]
        if task == "SegmentationPersonPart":
            urls.update(_PERSON_PART_URL)
        super().__init__(urls=urls, root=root)

    def move_to_raw_root(self):
        src_dir = f"{self.root}/VOCdevkit/VOC{self.year}"
        for directory in glob.glob(f"{src_dir}/*"):
            move(directory, self.raw_root)
        remove_directory(f"{self.root}/VOCdevkit")

    def _get_split_ids(self):
        task_dir = f"{self.raw_root}/ImageSets/{self.task}"
        id_file_suffix = "_id" if self.task == "SegmentationPersonPart" else ""
        train_ids = read_text_file(f"{task_dir}/train{id_file_suffix}.txt")
        val_ids = read_text_file(f"{task_dir}/val{id_file_suffix}.txt")
        if self.task == "Layout":
            train_ids = [_id.split(" ")[0] for _id in train_ids]
            val_ids = [_id.split(" ")[0] for _id in val_ids]
            train_ids = sorted(list(set(train_ids)))
            val_ids = sorted(list(set(val_ids)))
        return {"train": train_ids, "val": val_ids}

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

    def _get_filepaths(self, dirnames: list[str] = ["annots", "images", "labels"]):
        filepaths = {}
        for dirname in dirnames:
            splits_paths = {}
            for split in self.split_ids:
                paths = glob.glob(f"{self.task_root}/{dirname}/{split}/*")
                splits_paths[split] = sorted(paths)
            filepaths[dirname] = splits_paths
        return filepaths


class VOCMainDataProvider(VOCDataProvider):
    def __init__(self, root: str, labels_format: Literal["yolo"] = "yolo"):
        super().__init__(root=root, task="Main", labels_format=labels_format)


class VOCActionDataProvider(VOCDataProvider):
    def __init__(self, root: str, labels_format: Literal["yolo"] = "yolo"):
        super().__init__(root=root, task="Action", labels_format=labels_format)


class VOCLayoutDataProvider(VOCDataProvider):
    def __init__(self, root: str, labels_format: Literal["yolo"] = "yolo"):
        super().__init__(root=root, task="Layout", labels_format=labels_format)


class VOCSegmentationDataProvider(VOCDataProvider):
    def __init__(
        self,
        root: str,
        mode: Literal["semantic", "instance", "person_part"] = "semantic",
        labels_format: Literal["yolo"] = "yolo",
    ):
        self.mode = mode
        self.id2label = _SEG_ID2LABEL[mode]
        self.label2id = {v: k for k, v in self.id2label.items()}
        if mode == "semantic":
            mask_dirname = "SegmentationClass"
        elif mode == "instance":
            mask_dirname = "SegmentationObject"
        elif mode == "person_part":
            mask_dirname = "SegmentationPersonPart"
        self.mask_dirname = mask_dirname
        super().__init__(root=root, task=mask_dirname, labels_format=labels_format)

    def arrange_files(self):
        super().arrange_files()
        for split, ids in self.split_ids.items():
            src_masks_path = f"{self.raw_root}/{self.mask_dirname}"
            dst_masks_path = create_dir(Path(self.task_root) / "masks" / split)

            src_filepaths = [f"{src_masks_path}/{_id}.png" for _id in ids]
            dst_filepaths = [f"{dst_masks_path}/{_id}.png" for _id in ids]

            log.info(f"Moving {split} masks from {src_masks_path} to {dst_masks_path}")
            copy_files(src_filepaths, dst_filepaths)

    def _get_filepaths(self):
        return super()._get_filepaths(["annots", "images", "labels", "masks"])

    def create_labels(self):
        masks_filepaths = sorted(glob.glob(f"{self.task_root}/masks/*/*"))
        if self.labels_format == "yolo":
            parse_segmentation_masks_to_yolo(
                masks_filepaths,
                background=self.label2id["background"],
                contour=self.label2id["void"],
            )
        else:
            log.warn(f"Only YOLO label format is implemented ({self.labels_format} passed)")


class VOCInstanceSegmentationDataProvider(VOCSegmentationDataProvider):
    def __init__(self, root: str, labels_format: Literal["yolo"] = "yolo"):
        super().__init__(root=root, mode="instance", labels_format=labels_format)


class VOCSemanticSegmentationDataProvider(VOCSegmentationDataProvider):
    def __init__(self, root: str, labels_format: Literal["yolo"] = "yolo"):
        super().__init__(root=root, mode="semantic", labels_format=labels_format)


class VOCPersonPartSegmentationDataProvider(VOCSegmentationDataProvider):
    URL = "http://liangchiehchen.com/projects/DeepLab.html"

    def __init__(self, root: str, labels_format: Literal["yolo"] = "yolo"):
        super().__init__(root=root, mode="person_part", labels_format=labels_format)

    def unzip(self, remove: bool = False):
        Path(self.raw_root).mkdir(parents=True, exist_ok=True)
        is_voc_present = False
        is_personpart_present = False

        for zip_filepath in self.zip_filepaths:
            if zip_filepath.endswith("trainval_2012.tar") and path_exists(
                f"{self.raw_root}/Annotations"
            ):
                log.info("VOC trainval 2012 already downloaded. Skipping")
                is_voc_present = True
                continue
            elif zip_filepath.endswith("person_part.zip") and path_exists(
                f"{self.raw_root}/ImageSets/SegmentationPersonPart"
            ):
                is_personpart_present = True
                log.info("PersonPart dataset already downloaded. Skipping")
                continue
            else:
                unzip(zip_filepath, self.root, remove)
        self.move_to_raw_root(is_voc_present, is_personpart_present)
        if remove:
            self.zip_filepaths.clear()

    def move_to_raw_root(self, is_voc_present: bool, is_personpart_present: bool):
        if not is_voc_present:
            super().move_to_raw_root()
        if is_personpart_present:
            return
        ids_src_dir = f"{self.root}/pascal_person_part/pascal_person_part_trainval_list"
        ids_dst_dir = f"{self.raw_root}/ImageSets/SegmentationPersonPart"
        move(ids_src_dir, ids_dst_dir)

        masks_src_dir = f"{self.root}/pascal_person_part/pascal_person_part_gt"
        masks_dst_dir = f"{self.raw_root}/SegmentationPersonPart"
        move(masks_src_dir, masks_dst_dir)
        remove_directory(f"{self.root}/pascal_person_part")


if __name__ == "__main__":
    from geda.utils.config import ROOT

    root = str(ROOT / "data" / "voc_2012")
    # dp = VOCPersonPartSegmentationDataProvider(root)
    dp = VOCInstanceSegmentationDataProvider(root)
    # dp = VOCSemanticSegmentationDataProvider(root)
