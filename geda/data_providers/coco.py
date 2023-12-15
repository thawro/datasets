from geda.data_providers.base import DataProvider

from geda.utils.files import copy_files, save_yamls, create_dir, move_many

from pathlib import Path
from geda.utils.pylogger import get_pylogger
from dataclasses import dataclass
from PIL import Image
from typing import Optional
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import glob
from tqdm.auto import tqdm

log = get_pylogger(__name__)

_URLS = {
    "train2017": "http://images.cocodataset.org/zips/train2017.zip",
    "val2017": "http://images.cocodataset.org/zips/val2017.zip",
    "test2017": "http://images.cocodataset.org/zips/test2017.zip",
    "annotations_trainval2017": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
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

LIMBS = [
    (9, 7),
    (7, 5),
    (5, 3),
    (3, 1),
    (1, 0),
    (0, 2),
    (1, 2),
    (2, 4),
    (4, 6),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


@dataclass
class PointAnnotation:
    x: int
    y: int
    id: int
    visibility: bool

    @property
    def xy(self) -> tuple[int, int]:
        return self.x, self.y

    @classmethod
    def from_dict(cls, annot: dict) -> "PointAnnotation":
        return cls(**annot)


@dataclass
class PoseAnnotation:
    bbox: tuple[int, int, int, int]
    keypoints: dict[int, PointAnnotation]

    @classmethod
    def from_dict(cls, annot: dict):
        bbox = annot["bbox"]
        keypoints = {
            point["id"]: PointAnnotation.from_dict(point)
            for point in annot["keypoints"]
        }
        return cls(bbox, keypoints)


@dataclass
class Annotation:
    filename: str
    height: int
    width: int
    poses: Optional[list[PoseAnnotation]] | None = None

    @classmethod
    def from_dict(cls, annot: dict):
        filename = annot["filename"]
        height = annot["height"]
        width = annot["width"]
        poses = [PoseAnnotation.from_dict(obj_rect) for obj_rect in annot["objects"]]
        return cls(filename, height, width, poses)

    def plot(self):
        if self.poses is None:
            return
        image = np.array(Image.open(f"images/{self.filename}"))
        for pose in self.poses:
            keypoints = pose.keypoints
            for _, kp in keypoints.items():
                color = (0, 128, 255) if kp.visibility == 2 else (128, 128, 128)
                cv2.circle(image, kp.xy, 3, color, -1)
            for id_1, id_2 in LIMBS:
                if id_1 not in keypoints or id_2 not in keypoints:
                    continue
                kp_1 = keypoints[id_1]
                kp_2 = keypoints[id_2]

                cv2.line(image, kp_1.xy, kp_2.xy, (50, 255, 50), 4)
        image = cv2.resize(image, (0, 0), fx=2, fy=2)
        plt.figure(figsize=(12, 24))
        plt.imshow(image)


def parse_coco_keypoint_annot(annots: list[dict], info: dict) -> dict:
    annot_dict = info
    person_dicts = []
    for annot in annots:
        person_dict = {
            "bbox": annot["bbox"],
            "iscrowd": annot["iscrowd"],
            "num_keypoints": annot["num_keypoints"],
            "segmentation": annot["segmentation"],
        }
        kpts = annot["keypoints"]
        joint_dicts = []
        for i, (x, y, v) in enumerate(zip(kpts[::3], kpts[1::3], kpts[2::3])):
            joint_dict = {"x": int(x), "y": int(y), "id": i, "visibility": int(v)}
            joint_dicts.append(joint_dict)
        person_dict["keypoints"] = joint_dicts
        person_dicts.append(person_dict)
    annot_dict["objects"] = person_dicts
    return annot_dict


class COCOKeypointsDataProvider(DataProvider):
    URL = "https://cocodataset.org/#download"

    def __init__(self, root: str):
        self.task = "HumanPose"
        self.task_root = f"{root}/{self.task}"
        super().__init__(urls=_URLS, root=root)

    def move_to_raw_root(self):
        dirnames = ["train2017", "val2017", "test2017", "annotations"]
        src_paths = [f"{self.root}/{dirname}" for dirname in dirnames]
        dst_paths = [f"{self.raw_root}/{dirname}" for dirname in dirnames]
        move_many(src_paths, dst_paths)

    def _get_filepaths(self, dirnames: list[str] = ["annots", "images"]):
        super()._get_filepaths(dirnames)

    def set_annotations(self):
        annots_dir = f"{self.raw_root}/annotations"
        annot_splits = {"train": {}, "val": {}}
        for split in annot_splits:
            with open(f"{annots_dir}/person_keypoints_{split}2017.json") as f:
                annots = json.load(f)

            id2info = {}
            for info in annots["images"]:
                id2info[info["id"]] = {
                    "filename": info["file_name"],
                    "height": info["height"],
                    "width": info["width"],
                }

            id2annots = {}

            for annot in annots["annotations"]:
                img_id = annot["image_id"]
                if img_id not in id2annots:
                    id2annots[img_id] = [annot]
                else:
                    id2annots[img_id].append(annot)

            for _id, annots in id2annots.items():
                id2annots[_id] = parse_coco_keypoint_annot(annots, id2info[_id])
            annot_splits[split] = id2annots
        test_filepaths = sorted(glob.glob(f"{self.raw_root}/test2017/*"))
        test_id2annots = {}
        for filepath in test_filepaths:
            filename = filepath.split("/")[-1]
            _id = filename.split(".")[0]
            image = Image.open(filepath)
            test_id2annots[_id] = {
                "filename": filename,
                "height": image.height,
                "width": image.width,
            }
        annot_splits["test"] = test_id2annots
        self.annots = annot_splits

    def _get_split_ids(self):
        self.set_annotations()

        return {
            split: list(split_annots.keys())
            for split, split_annots in self.annots.items()
        }

    def arrange_files(self):
        Path(self.task_root).mkdir(parents=True, exist_ok=True)
        for split, annots in tqdm(self.annots.items(), desc="Splits"):
            names = ["annots", "images"]
            dst_paths = {
                name: create_dir(Path(self.task_root) / name / split) for name in names
            }
            src_imgs_path = f"{self.raw_root}/{split}2017"
            dst_imgs_path = dst_paths["images"]

            dst_annots_path = dst_paths["annots"]

            filenames = [annot["filename"] for _id, annot in annots.items()]
            src_imgs_filepaths = [f"{src_imgs_path}/{fname}" for fname in filenames]
            dst_imgs_filepaths = [f"{dst_imgs_path}/{fname}" for fname in filenames]

            log.info(f"Copying {split} images from {src_imgs_path} to {dst_imgs_path}")
            copy_files(src_imgs_filepaths, dst_imgs_filepaths)

            log.info(f"Saving {split} annotations as .yaml files in {dst_annots_path}")
            yaml_paths = [
                f'{dst_annots_path}/{annot["filename"].replace(".jpg", "")}.yaml'
                for annot in annots.values()
            ]
            save_yamls(list(annots.values()), yaml_paths)


if __name__ == "__main__":
    from geda.utils.config import ROOT

    root = str(ROOT / "data" / "COCO")
    dp = COCOKeypointsDataProvider(root)
    dp.get_data()
