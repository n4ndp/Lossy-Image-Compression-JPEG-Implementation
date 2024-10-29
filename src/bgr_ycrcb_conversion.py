import math
import numpy as np

def bgr_image_to_ycrcb(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)

    B = image[:, :, 0]
    G = image[:, :, 1]
    R = image[:, :, 2]

    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 128
    Cb = (B - Y) * 0.564 + 128

    ycrcb_image = np.stack((Y, Cr, Cb), axis=-1)

    return np.round(ycrcb_image).astype(np.uint8)

def ycrcb_image_to_bgr(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)

    Y = image[:, :, 0]
    Cr = image[:, :, 1] - 128
    Cb = image[:, :, 2] - 128

    R = Y + 1.403 * Cr
    G = Y - 0.344 * Cb - 0.714 * Cr
    B = Y + 1.770 * Cb

    bgr_image = np.stack((B, G, R), axis=-1)
    bgr_image = np.clip(bgr_image, 0, 255)

    return np.round(bgr_image).astype(np.uint8)
