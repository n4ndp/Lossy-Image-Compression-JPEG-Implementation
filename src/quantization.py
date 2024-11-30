import numpy as np

QTY = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

QTC = np.array([
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
    """
    Escala una matriz de cuantización en función de la calidad deseada.
    
    Args:
        quant_matrix (np.ndarray): Matriz de cuantización a escalar.
        quality (int): Calidad deseada.
        
    Returns:
        np.ndarray: Matriz de cuantización escalada.
    """
    if quality < 1 or quality > 100:
        raise ValueError("La calidad debe estar entre 1 y 100.")
    
    scaling_factor = 50 / quality if quality < 50 else 2 - quality / 50
    scaled_quant_matrix = quant_matrix * scaling_factor
    
    return np.clip(scaled_quant_matrix, 1, 255).astype(np.uint8)

def quantize(channel: np.ndarray, quant_matrix: np.ndarray, inverse: bool = False) -> np.ndarray:
    """
    Aplica la cuantización o des-cuantización a un canal de la imagen.
    
    Args:
        channel (np.ndarray): Canal de la imagen a cuantizar o des-cuantizar.
        quant_matrix (np.ndarray): Matriz de cuantización a utilizar.
        inverse (bool, opcional): Si es `True`, se des-cuantiza, 
                                 si es `False`, se cuantiza. Por defecto es `False`.

    Returns:
        np.ndarray: Canal cuantizado
    """
    if channel is None or channel.size == 0:
        raise ValueError("El canal de entrada está vacío.")
    
    if quant_matrix is None or quant_matrix.size == 0:
        raise ValueError("La matriz de cuantización está vacía.")
    
    if channel.shape[0] % 8 != 0 or channel.shape[1] % 8 != 0:
        raise ValueError("Las dimensiones del canal deben ser múltiplos de 8.")

    n = 8
    height, width = channel.shape
    quantized_channel = np.zeros_like(channel, dtype=np.float32)

    for y in range(0, height, n):
        for x in range(0, width, n):
            block = channel[y:y+n, x:x+n]

            if inverse:
                quantized_channel[y:y+n, x:x+n] = np.round(block * quant_matrix)
            else:
                quantized_channel[y:y+n, x:x+n] = np.round(block / quant_matrix)
                
    return quantized_channel.astype(np.int16)
