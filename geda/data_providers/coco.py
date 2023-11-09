from geda.data_providers.base import DataProvider

from geda.utils.files import copy_files, save_yaml, create_dir, move
from pathlib import Path
from geda.utils.pylogger import get_pylogger
import scipy
from dataclasses import dataclass
from PIL import Image
from typing import Optional
import numpy as np
import cv2
import matplotlib.pyplot as plt

log = get_pylogger(__name__)

_URLS = {
    "train_data_2017": "http://images.cocodataset.org/zips/train2017.zip",
    "val_data_2017": "http://images.cocodataset.org/zips/val2017.zip",
    "test_data_2017": "http://images.cocodataset.org/zips/test2017.zip",
    "train_val_annots_2017": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}

LABELS = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

LIMBS = []


@dataclass
class PointAnnotation:
    x: int
    y: int
    id: int
    is_visible: bool

    @property
    def xy(self) -> tuple[int, int]:
        return self.x, self.y

    @classmethod
    def from_dict(cls, annot: dict) -> "PointAnnotation":
        return cls(**annot)


@dataclass
class PoseAnnotation:
    scale: float
    objpos_xy: tuple[int, int]
    head_xyxy: tuple[int, int, int, int]
    keypoints: list[PointAnnotation]

    @classmethod
    def from_dict(cls, annot: dict):
        scale = annot["scale"]
        objpos_xy = annot["objpos_xy"]
        head_xyxy = annot["head_xyxy"]
        keypoints = {
            point["id"]: PointAnnotation.from_dict(point)
            for point in annot["keypoints"]
        }
        return cls(scale, objpos_xy, head_xyxy, keypoints)


@dataclass
class Annotation:
    filename: str
    is_valid: bool
    poses: Optional[list[PoseAnnotation]] | None = None

    @classmethod
    def from_dict(cls, annot: dict):
        filename = annot["filename"]
        is_valid = annot["is_valid"]
        if not is_valid:
            return cls(filename, is_valid)
        poses = [PoseAnnotation.from_dict(obj_rect) for obj_rect in annot["objects"]]
        return cls(filename, is_valid, poses)

    def plot(self):
        if not self.is_valid or self.poses is None:
            return
        image = np.array(Image.open(f"images/{self.filename}"))
        for pose in self.poses:
            keypoints = pose.keypoints
            for _, kp in keypoints.items():
                color = (0, 128, 255) if kp.is_visible else (128, 128, 128)
                cv2.circle(image, kp.xy, 3, color, -1)
            for id_1, id_2 in LIMBS:
                if id_1 not in keypoints or id_2 not in keypoints:
                    continue
                kp_1 = keypoints[id_1]
                kp_2 = keypoints[id_2]
                cv2.line(image, kp_1.xy, kp_2.xy, (50, 255, 50), 4)
        plt.imshow(image)


def parse_coco_annotation(annot) -> dict:
    """
    returns annot dict in form
    {

    }
    """

    annot_dict = {}
    return annot_dict


class COCOKeypointsDataProvider(DataProvider):
    URL = "https://cocodataset.org/#download"

    def __init__(self, root: str):
        self.task = "HumanPose"
        self.task_root = f"{root}/{self.task}"
        super().__init__(urls=_URLS, root=root)

    def move_to_raw_root(self):
        src_annots_path = f"{self.root}/annots"
        src_data_path = f"{self.root}/images"
        move(src_annots_path, f"{self.raw_root}/annots")
        move(src_data_path, f"{self.raw_root}/images")

    def _get_filepaths(self, dirnames: list[str] = ["annots", "images"]):
        super()._get_filepaths(dirnames)

    def _get_split_ids(self):
        test_idxs = None
        train_idxs = None
        val_idxs = None

        return {
            "train": train_idxs,
            "val": val_idxs,
            "test": test_idxs,
        }

    def arrange_files(self):
        Path(self.task_root).mkdir(parents=True, exist_ok=True)

        for split, idxs in self.split_ids.items():
            names = ["annots", "images"]
            dst_paths = {
                name: create_dir(Path(self.task_root) / name / split) for name in names
            }
            src_imgs_path = f"{self.raw_root}/images"
            dst_imgs_path = dst_paths["images"]
            dst_annots_path = dst_paths["annots"]

            filenames = [annot["filename"] for annot in split_annots]
            src_imgs_filepaths = [f"{src_imgs_path}/{fname}" for fname in filenames]
            dst_imgs_filepaths = [f"{dst_imgs_path}/{fname}" for fname in filenames]

            log.info(f"Moving {split} images from {src_imgs_path} to {dst_imgs_path}")
            copy_files(src_imgs_filepaths, dst_imgs_filepaths)

            log.info(f"Saving {split} annotations as .yaml files in {dst_annots_path}")
            for annot in split_annots:
                _id = annot["filename"].replace(".jpg", "")
                save_yaml(annot, f"{dst_annots_path}/{_id}.yaml")


if __name__ == "__main__":
    from geda.utils.config import ROOT

    root = str(ROOT / "data" / "COCO")
    dp = COCOKeypointsDataProvider(root)
    dp.get_data()
