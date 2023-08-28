from geda.utils.colors import color_map
import png
import numpy as np
import math


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
