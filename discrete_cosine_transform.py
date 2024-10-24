import math
import cv2
import numpy as np
from bgr_ycrcb_conversion import bgr_image_to_ycrcb, ycrcb_image_to_bgr
from functions import pad, unpad, downsampling, upsampling

def matrix_C(n: int) -> np.ndarray:
    """Genera la matriz de coeficientes C para la Transformada Discreta del Coseno (DCT).
    
    Args:
        n (int): Dimensión de la matriz C (n x n).

    Returns:
        np.ndarray: Matriz C de tamaño n x n usada para DCT o IDCT.
    """
    C = np.zeros((n, n), dtype=np.float32)
    
    for i in range(n):
        if i == 0:
            alpha = 1 / math.sqrt(n)
        else:
            alpha = math.sqrt(2/n)
        
        for j in range(n):
            C[i][j] = alpha * math.cos((2*j + 1) * i * math.pi / (2*n))
    
    return C

def discrete_cosine_transform(channel: np.ndarray, inverse: bool = False) -> np.ndarray:
    """Aplica la Transformada Discreta del Coseno (DCT) o su inversa (IDCT) a un canal de imagen.

    Esta función divide el canal de imagen en bloques de 8x8 y aplica la DCT o 
    la IDCT (si inverse=True) sobre cada bloque. La DCT convierte la imagen al dominio 
    de la frecuencia, mientras que la IDCT reconstruye la imagen desde el dominio de 
    la frecuencia.

    Args:
        channel (np.ndarray): El canal de imagen que será transformado.
        inverse (bool): Si es True, se aplica la Inversa de la DCT (IDCT).

    Returns:
        np.ndarray: El canal transformado (después de aplicar DCT o IDCT).
    """
    C = matrix_C(8)
    CT = C.T
    
    height, width = channel.shape
    
    if height % 8 != 0 or width % 8 != 0:
        raise ValueError('Channel dimensions must be multiples of 8.')
    
    transformed_channel = np.zeros_like(channel, dtype=np.float32)
    
    for y in range(0, height, 8):
        for x in range(0, width, 8):
            tile = channel[y:y+8, x:x+8]
            
            if inverse:
                transformed_tile = CT @ tile @ C
            else:
                transformed_tile = C @ tile @ CT
                
            transformed_channel[y:y+8, x:x+8] = transformed_tile
            
    return transformed_channel

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
    
    Y_restored = discrete_cosine_transform(Y_transformed, inverse=True)
    Cr_restored = discrete_cosine_transform(Cr_transformed, inverse=True)
    Cb_restored = discrete_cosine_transform(Cb_transformed, inverse=True)
    
    Y_restored = unpad(Y_restored, Y.shape[0], Y.shape[1])
    Cr_restored = unpad(Cr_restored, Cr_downsampled.shape[0], Cr_downsampled.shape[1])
    Cb_restored = unpad(Cb_restored, Cb_downsampled.shape[0], Cb_downsampled.shape[1])
    
    ycrcb_image_restored = upsampling(Y_restored, Cr_restored, Cb_restored)
    
    bgr_image_restored = ycrcb_image_to_bgr(ycrcb_image_restored)
    
    cv2.imshow('Original Image', image)
    cv2.imshow('Restored Image', bgr_image_restored)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
