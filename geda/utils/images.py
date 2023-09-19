from geda.utils.colors import color_map
import png
import numpy as np
import math
from PIL import Image


def load_img_to_array(filepath: str) -> np.ndarray:
    if filepath.endswith((".jpg", "png", "jpeg")):
        return np.array(Image.open(filepath))
    elif filepath.endswith(".npy"):
        return np.load(filepath)
    else:
        raise Exception("Wrong file extension. .jpg, .png, .jpeg and .npy are supported.")


def save_gray_array_as_color_png(
    array: np.ndarray, palette: list[tuple[int, int, int]] | None, filename: str = "gray2color.png"
):
    height, width = array.shape
    bitdepth = math.ceil(math.log2(len(palette)))
    if palette is None:
        palette = color_map(256)
    w = png.Writer(width, height, palette=palette, bitdepth=bitdepth)
    f = open(filename, "wb")
    w.write(f, array.tolist())
