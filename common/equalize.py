import cv2


def clahe_equalize_bgr(img_bgr, clip_limit=2.5, tile_grid=(8, 8)):
    """Contrast Limited Adaptive Histogram Equalization, applied only to
    the L (lightness) channel in LAB color space.

    Fixes uneven/dark lighting locally without distorting color (unlike
    equalizing each BGR channel separately, which shifts hue/saturation)
    and without the harsh, noise-amplified look of a single global
    histogram equalization -- the clip limit caps how much any one
    tile's contrast can be stretched, and each tile is equalized against
    its own local neighborhood rather than the whole image's histogram.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_eq = clahe.apply(l)
    merged = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
