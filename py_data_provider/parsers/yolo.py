from PIL import Image
import numpy as np
import cv2
import glob
from tqdm.auto import tqdm


def find_polygons_in_binary_mask(binary_mask: np.ndarray) -> list[np.ndarray]:
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    polygons = [polygon.squeeze() for polygon in contours]
    return polygons


def parse_segmentation_mask_to_yolo_format(
    mask: np.ndarray,
    background: int | None = None,
    contour: int | None = None,
    return_str: bool = True,
) -> list[list[int | float]] | str:
    """Return list of YOLO like annotations for segmentation purposes, i.e.
    [
        [class_id, x1, y1, x2, y2, ..., xn, yn], # object_0
        [class_id, x1, y1, x2, y2, ..., xn, yn], # object_1
        ...,
        [class_id, x1, y1, x2, y2, ..., xn, yn] # object_m
    ]
    mask attribute is a
    """
    unique_label_ids = np.unique(mask).tolist()
    if background is not None and background in unique_label_ids:
        unique_label_ids.remove(background)  # remove background from labels
    if contour is not None and contour in unique_label_ids:
        unique_label_ids.remove(contour)  # remove contour from labels
    annotations = []
    for label_id in unique_label_ids:
        binary_mask = ((mask == label_id) * 1).astype(np.uint8)
        polygons = find_polygons_in_binary_mask(binary_mask)
        wh = np.flip(np.array(binary_mask.shape))  # for normalization purposes
        norm_polygons = [polygon / wh for polygon in polygons]
        xy_sequences = [polygon.flatten().tolist() for polygon in norm_polygons]
        for xy_sequence in xy_sequences:
            if len(xy_sequence) // 2 < 3:
                continue
            annotations.append([label_id] + xy_sequence)
    if return_str:
        annotations_str = [" ".join([str(el) for el in annot]) for annot in annotations]
        return "\n".join(annotations_str)
    return annotations


def parse_segmentation_masks_to_yolo(
    masks_filepaths: list[str],
    background: int | None = None,
    contour: int | None = None,
):
    for filename in tqdm(masks_filepaths, desc="Creating labels in yolo format"):
        mask = np.array(Image.open(filename))
        label_filepath = filename.replace("masks", "labels").replace(".png", ".txt")
        annot_txt = parse_segmentation_mask_to_yolo_format(
            mask=mask, background=background, contour=contour
        )
        with open(label_filepath, "w") as file:
            file.write(annot_txt)
