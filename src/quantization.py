import math
import numpy as np

quant_matrix_Y = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

quant_matrix_C = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
])

def scale_quant_matrix(quant_matrix: np.ndarray, quality: int) -> np.ndarray:
    if quality < 1: quality = 1
    if quality > 100: quality = 100

    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - (quality * 2)

    scaled_quant_matrix = np.floor((quant_matrix * scale + 50) / 100)
    scaled_quant_matrix[scaled_quant_matrix == 0] = 1
    
    return scaled_quant_matrix

def quantize(channel: np.ndarray, quant_matrix: np.ndarray, inverse: bool = False) -> np.ndarray:
    height, width = channel.shape
    
    if height % 8 != 0 or width % 8 != 0:
        raise ValueError('Channel dimensions must be multiples of 8.')
    
    quantized_channel = np.zeros_like(channel, dtype=np.float32)
    
    for y in range(0, height, 8):
        for x in range(0, width, 8):
            tile = channel[y:y+8, x:x+8]
            
            if inverse:
                quantized_tile = tile * quant_matrix
            else:
                quantized_tile = np.round(tile / quant_matrix)
                
            quantized_channel[y:y+8, x:x+8] = quantized_tile
            
    return quantized_channel
