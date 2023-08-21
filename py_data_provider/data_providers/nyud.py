from py_data_provider.data_providers.base import DataProvider
from py_data_provider.utils.files import (
    path_exists,
    move,
    save_txt_to_file,
    create_dir,
)
from py_data_provider.parsers.yolo import parse_segmentation_masks_to_yolo
from pathlib import Path
import glob
from py_data_provider.utils.pylogger import get_pylogger
import scipy
import mat73
from PIL import Image
from typing import Literal
import numpy as np
from tqdm.auto import tqdm

log = get_pylogger(__name__)


_URLS = {
    "nyu_depth_v2_labeled": "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat",
    "splits": "http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat",
}


class NYUDv2DataProvider(DataProvider):
    URL = "https://cs.nyu.edu/~silberman/projects/indoor_scene_seg_sup.html"

    def __init__(self, root: str, labels_format: Literal["yolo"] = "yolo"):
        self.splits = ["train", "test"]
        self.task = "Segmentation"
        self.task_root = f"{root}/{self.task}"
        self.labels_format = labels_format
        super().__init__(urls=_URLS, root=root)

    def check_if_present(self) -> bool:
        return path_exists(self.task_root)

    def move_to_raw_root(self):
        src_data_path = f"{self.root}/nyu_depth_v2_labeled.mat"
        src_splits_path = f"{self.root}/splits.mat"
        # dst_data_path = f"{self.raw_root}/nyud_depth_v2_labeled.mat"
        # dst_splits_path = f"{self.raw_root}/splits.mat"
        move(src_data_path, self.raw_root)
        move(src_splits_path, self.raw_root)

    def _get_split_ids(self):
        splits = scipy.io.loadmat(f"{self.raw_root}/splits.mat")
        train_idxs = splits["trainNdxs"].flatten() - 1
        test_idxs = splits["testNdxs"].flatten() - 1
        return {
            "train": train_idxs.tolist(),
            "test": test_idxs.tolist(),
        }

    def arrange_files(self):
        log.info("Loading .mat file with data. It may take a while..")
        data_dict = mat73.loadmat(f"{self.raw_root}/nyu_depth_v2_labeled.mat")
        # H=480, W=640, N=1449
        numpy_images_data = {  # images to be saved as .npy files
            "semantic_masks": data_dict["labels"],  # HWN
            "instance_masks": data_dict["instances"],  # HWN
            "depth_masks": data_dict["depths"],  # HWN
        }

        images = data_dict["images"]  # HW3N

        classnames = ["unlabeled"] + [names[0] for names in data_dict["names"]]  # 1 + 894
        scenes = [names[0] for names in data_dict["sceneTypes"]]  # N

        classnames_txt = "\n".join(classnames)
        Path(self.task_root).mkdir(parents=True, exist_ok=True)
        save_txt_to_file(classnames_txt, f"{self.task_root}/classnames.txt")

        for split, idxs in self.split_ids.items():
            names = list(numpy_images_data.keys()) + ["labels", "scenes", "images"]
            paths = {name: create_dir(Path(self.task_root) / name / split) for name in names}
            for name in names:
                log.info(f"Saving {split} {name} to {paths[name]}")

            for idx in tqdm(idxs, f"{split} saving images"):
                for name, image_data in numpy_images_data.items():
                    img = image_data[..., idx]
                    np.save(f"{paths[name]}/{idx}.npy", img)

                Image.fromarray(images[..., idx]).save(f"{paths['images']}/{idx}.jpg")
                scene = scenes[idx]
                scene_filepath = f"{paths['scenes']}/{idx}.txt"
                save_txt_to_file(scene, scene_filepath)

    def _get_filepaths(self, dirnames: list[str] = ["masks", "images", "labels"]):
        filepaths = {}
        for dirname in dirnames:
            splits_paths = {}
            for split in self.split_ids:
                paths = glob.glob(f"{self.task_root}/{dirname}/{split}/*")
                splits_paths[split] = sorted(paths)
            filepaths[dirname] = splits_paths
        return filepaths

    def create_labels(self):
        masks_filepaths = sorted(glob.glob(f"{self.task_root}/masks/*/*"))
        if self.labels_format == "yolo":
            parse_segmentation_masks_to_yolo(masks_filepaths, background=0, contour=None)
        else:
            log.warn(f"Only YOLO label format is implemented ({self.labels_format} passed)")


if __name__ == "__main__":
    from py_data_provider.utils.config import ROOT

    root = str(ROOT / "data" / "NYUDv2")
    dp = NYUDv2DataProvider(root)
