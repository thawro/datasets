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
    "images": "https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz",
    "annotations": "https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1_u12_2.zip",
}

LABELS = [
    "r_ankle",
    "r_knee",
    "r_hip",
    "l_hip",
    "l_knee",
    "l_ankle",
    "pelvis",
    "thorax",
    "upper_neck",
    "head_top",
    "r_wrist",
    "r_elbow",
    "r_shoulder",
    "l_shoulder",
    "l_elbosr",
    "l_wrist",
]

LIMBS = [
    [0, 1],  # R_ankle - R_knee
    [1, 2],  # R_knee - R_hip
    [2, 6],  # R_hip - pelvis
    [5, 4],  # L_ankle - L_knee
    [4, 3],  # L_knee - L_hip
    [3, 6],  # L_hip - pelvis
    [6, 7],  # pelvis - thorax
    [7, 8],  # thorax - upper_neck
    [8, 9],  # upper_neck - head_top
    [8, 12],  # upper_neck - R_shoulder
    [12, 11],  # R_shoulder - R_elbow
    [11, 10],  # R_elbow - R_wrist
    [8, 13],  # upper_neck - L_shoulder
    [13, 14],  # L_shoulder - L_elbow
    [14, 15],  # L_elbow - L_wrist
]


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


def parse_mpii_annotation(annot: np.ndarray) -> dict:
    """
    returns annot dict in form
    {
        'filename': string,
        'is_valid': bool, # False when there are no joint annotations
        'annorect': [
            {
                'scale': float,
                'objpos_xy': [int, int],
                'head_xyxy': [int, int, int, int],
                'keypoints': [
                    {
                        'x': int,
                        'y': int,
                        'id': int,
                        'is_visible': bool
                    },
                    ...
                ]
            },
            ...
        ]
    }

    """
    filename = annot["image"]["name"].item()[0].item()
    annorects = annot["annorect"]
    if len(annorects) == 0:
        is_valid = False
    else:
        annorects = annorects[0]
        is_valid = (
            annorects.dtype.names is not None and "annopoints" in annorects.dtype.names
        )

    annot_dict = {"filename": filename, "is_valid": is_valid}
    if not is_valid:
        return annot_dict

    person_dicts = []
    for ridx, annorect in enumerate(annorects):
        try:
            scale = annorect["scale"][0, 0].item()
        except IndexError:
            continue
        objpos_xy = [pos[0, 0].item() for pos in annorect["objpos"][0, 0]]
        head_xyxy = [annorect[coord][0, 0].item() for coord in ["x1", "y1", "x2", "y2"]]

        person_dict = {"scale": scale, "objpos_xy": objpos_xy, "head_xyxy": head_xyxy}

        annopoints = annorect["annopoints"]["point"][0, 0][0]

        joints_dicts = []
        for point in annopoints:
            is_visible = (
                "is_visible" in point.dtype.names
                and [1] in point["is_visible"].tolist()
            )
            joint_dict = {
                "x": int(point["x"][0, 0].item()),
                "y": int(point["y"][0, 0].item()),
                "id": point["id"][0, 0].item(),
                "is_visible": is_visible,
            }
            joints_dicts.append(joint_dict)

        person_dict["keypoints"] = joints_dicts
        person_dicts.append(person_dict)
    annot_dict["objects"] = person_dicts
    return annot_dict


class MPIIDataProvider(DataProvider):
    URL = "http://human-pose.mpi-inf.mpg.de/"

    def __init__(self, root: str):
        self.task = "HumanPose"
        self.task_root = f"{root}/{self.task}"
        super().__init__(urls=_URLS, root=root)

    def move_to_raw_root(self):
        src_annots_path = f"{self.root}/mpii_human_pose_v1_u12_2"
        src_data_path = f"{self.root}/images"
        move(src_annots_path, f"{self.raw_root}/annots")
        move(src_data_path, f"{self.raw_root}/images")

    def _get_filepaths(self, dirnames: list[str] = ["annots", "images"]):
        super()._get_filepaths(dirnames)

    def set_annotations(self):
        annots_struct = scipy.io.loadmat(
            f"{self.raw_root}/annots/mpii_human_pose_v1_u12_1.mat"
        )
        release = annots_struct["RELEASE"][0, 0]
        annolist = release["annolist"][0]
        # train/val/test splits are determined on annotations validity, not on "img_train" field
        is_train = release["img_train"][0]  # not used
        annots = [parse_mpii_annotation(a) for a in annolist]
        self.annots = annots

    def _get_split_ids(self):
        self.set_annotations()
        valid = [a["is_valid"] for a in self.annots]

        train_val_idxs = np.where(valid)[0]
        test_idxs = np.where(~np.array(valid))[0]

        train_ratio = 0.8
        train_n_samples = int(train_ratio * len(train_val_idxs))
        train_idxs = train_val_idxs[:train_n_samples]
        val_idxs = train_val_idxs[train_n_samples:]

        return {
            "train": train_idxs.tolist(),
            "val": val_idxs.tolist(),
            "test": test_idxs.tolist(),
        }

    def arrange_files(self):
        Path(self.task_root).mkdir(parents=True, exist_ok=True)

        for split, idxs in self.split_ids.items():
            split_annots = [self.annots[idx] for idx in idxs]
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

    root = str(ROOT / "data" / "MPII")
    dp = MPIIDataProvider(root)
    dp.get_data()
