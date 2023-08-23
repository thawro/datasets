from geda.utils.colors import color_map
import png
import numpy as np


def save_gray_array_as_color_png(
    array: np.ndarray, n_colors: int = 256, filename: str = "gray2color.png"
):
    height, width = array.shape
    palette = color_map(n_colors)
    w = png.Writer(width, height, palette=palette, bitdepth=8)
    f = open(filename, "wb")
    w.write(f, array.tolist())
