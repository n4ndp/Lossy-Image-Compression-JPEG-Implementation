import math
import numpy as np

def pad(channel: np.ndarray) -> np.ndarray:
    height, width = channel.shape

    height_padding = math.ceil(height / 8) * 8 - height
    width_padding = math.ceil(width / 8) * 8 - width

    padded_channel = np.pad(channel, pad_width=((0, height_padding), (0, width_padding)), mode='constant')

    return padded_channel

def unpad(channel: np.ndarray, original_height: int, original_width: int) -> np.ndarray:
    return channel[:original_height, :original_width]

def downsampling(ycrcb_image: np.ndarray) -> tuple:
    Y = ycrcb_image[:, :, 0]
    Cr = ycrcb_image[:, :, 1]
    Cb = ycrcb_image[:, :, 2]

    Cr_downsampled = Cr[::2, ::2]
    Cb_downsampled = Cb[::2, ::2]

    return Y, Cr_downsampled, Cb_downsampled

def upsampling(Y: np.ndarray, Cr_downsampled: np.ndarray, Cb_downsampled: np.ndarray) -> np.ndarray:
    Cr = Cr_downsampled.repeat(2, axis=0).repeat(2, axis=1)
    Cb = Cb_downsampled.repeat(2, axis=0).repeat(2, axis=1)

    Cr = Cr[:Y.shape[0], :Y.shape[1]]
    Cb = Cb[:Y.shape[0], :Y.shape[1]]

    ycrcb_image = np.stack((Y, Cr, Cb), axis=-1)

    return ycrcb_image
