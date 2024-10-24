import math
import cv2
import numpy as np
from bgr_ycrcb_conversion import bgr_image_to_ycrcb, ycrcb_image_to_bgr
from functions import pad, unpad, downsampling, upsampling
from discrete_cosine_transform import discrete_cosine_transform

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
    """Escala la matriz de cuantización según el factor de calidad.
    
    Args:
        quant_matrix (np.ndarray): La matriz de cuantización base (8x8).
        quality (int): Factor de calidad entre 1 y 100 (100 = mejor calidad).

    Returns:
        np.ndarray: Matriz de cuantización escalada según la calidad.
    """
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
    """Aplica la cuantización o dequantización a un canal de imagen.
        
    Esta función divide el canal de imagen en bloques de 8x8 y aplica la cuantización
    o dequantización (si inverse=True) sobre cada bloque. La cuantización reduce la
    precisión de los coeficientes de la DCT, mientras que la dequantización restaura
    la precisión de los coeficientes.
    
    Args:
        channel (np.ndarray): El canal de imagen que será cuantizado o dequantizado.
        quant_matrix (np.ndarray): La matriz de cuantización a aplicar.
        inverse (bool): Si es True, se aplica la dequantización.
        
    Returns:
        np.ndarray: El canal cuantizado o dequantizado.
    """
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


if __name__ == "__main__":
    image = cv2.imread('img/lena.bmp')
    
    ycrcb_image = bgr_image_to_ycrcb(image)
    
    Y, Cr_downsampled, Cb_downsampled = downsampling(ycrcb_image)
    
    Y_padded = pad(Y)
    Cr_downsampled_padded = pad(Cr_downsampled)
    Cb_downsampled_padded = pad(Cb_downsampled)
    
    Y_transformed = discrete_cosine_transform(Y_padded)
    Cr_transformed = discrete_cosine_transform(Cr_downsampled_padded)
    Cb_transformed = discrete_cosine_transform(Cb_downsampled_padded)
    
    quality = 25
    scaled_quant_matrix_Y = scale_quant_matrix(quant_matrix_Y, quality)
    scaled_quant_matrix_C = scale_quant_matrix(quant_matrix_C, quality)

    Y_quantized = quantize(Y_transformed, scaled_quant_matrix_Y)
    Cr_quantized = quantize(Cr_transformed, scaled_quant_matrix_C)
    Cb_quantized = quantize(Cb_transformed, scaled_quant_matrix_C)

    Y_dequantized = quantize(Y_quantized, scaled_quant_matrix_Y, inverse=True)
    Cr_dequantized = quantize(Cr_quantized, scaled_quant_matrix_C, inverse=True)
    Cb_dequantized = quantize(Cb_quantized, scaled_quant_matrix_C, inverse=True)

    Y_restored = discrete_cosine_transform(Y_dequantized, inverse=True)
    Cr_restored = discrete_cosine_transform(Cr_dequantized, inverse=True)
    Cb_restored = discrete_cosine_transform(Cb_dequantized, inverse=True)
    
    Y_restored = unpad(Y_restored, Y.shape[0], Y.shape[1])
    Cr_restored = unpad(Cr_restored, Cr_downsampled.shape[0], Cr_downsampled.shape[1])
    Cb_restored = unpad(Cb_restored, Cb_downsampled.shape[0], Cb_downsampled.shape[1])
    
    ycrcb_image_restored = upsampling(Y_restored, Cr_restored, Cb_restored)
    
    bgr_image_restored = ycrcb_image_to_bgr(ycrcb_image_restored)
    
    cv2.imshow('Original Image', image)
    cv2.imshow('Restored Image', bgr_image_restored)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
