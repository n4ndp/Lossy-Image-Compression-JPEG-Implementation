import math
import numpy as np

def pad(channel: np.ndarray) -> np.ndarray:
    height, width = channel.shape
    
    if height % 8 == 0 and width % 8 == 0:
        return channel

    height_padding = (8 - height % 8) % 8
    width_padding = (8 - width % 8) % 8

    padded_channel = np.pad(channel, ((0, height_padding), (0, width_padding)), mode='constant')

    return padded_channel

def unpad(channel: np.ndarray, original_height: int, original_width: int) -> np.ndarray:
    height, width = channel.shape
    
    if height == original_height and width == original_width:
        return channel
    
    unpadded_channel = channel[:original_height, :original_width]
    
    return unpadded_channel

def downsampling(ycrcb_image: np.ndarray) -> tuple:
    Y = ycrcb_image[:, :, 0]
    Cr_downsampled = ycrcb_image[:, :, 1][::2, ::2]
    Cb_downsampled = ycrcb_image[:, :, 2][::2, ::2]

    return Y, Cr_downsampled, Cb_downsampled

def upsampling(Y: np.ndarray, Cr_downsampled: np.ndarray, Cb_downsampled: np.ndarray) -> np.ndarray:
    Cr = np.repeat(np.repeat(Cr_downsampled, 2, axis=0), 2, axis=1)
    Cb = np.repeat(np.repeat(Cb_downsampled, 2, axis=0), 2, axis=1)
    
    Cr = Cr[:Y.shape[0], :Y.shape[1]]
    Cb = Cb[:Y.shape[0], :Y.shape[1]]
    
    ycrcb_image = np.stack((Y, Cr, Cb), axis=-1)

    return ycrcb_image
