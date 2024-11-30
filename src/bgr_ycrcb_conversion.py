import numpy as np

def bgr_image_to_ycbcr(image: np.ndarray) -> np.ndarray:
    """
    Convierte una imagen en formato BGR a formato YCbCr.
    
    Args:
        image (np.ndarray): Imagen en formato BGR.
        
    Returns:
        np.ndarray: Imagen en formato YCbCr.
    """
    if image is None or image.size == 0:
        raise ValueError("La imagen de entrada está vacía.")
    
    image = image.astype(np.float32)
    B, G, R = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    
    Y  =  0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.168736 * R - 0.331264 * G + 0.5 * B + 128
    Cr =  0.5 * R - 0.418688 * G - 0.081312 * B + 128
    
    return np.clip(np.stack((Y, Cb, Cr), axis=-1), 0, 255).astype(np.uint8)

def ycbcr_image_to_bgr(image: np.ndarray) -> np.ndarray:
    """
    Convierte una imagen en formato YCbCr a formato BGR.
    
    Args:
        image (np.ndarray): Imagen en formato YCbCr.
        
    Returns:
        np.ndarray: Imagen en formato BGR.
    """
    if image is None or image.size == 0:
        raise ValueError("La imagen de entrada está vacía.")
    
    image = image.astype(np.float32)
    Y, Cb, Cr = image[:, :, 0], image[:, :, 1] - 128, image[:, :, 2] - 128
    
    R = Y + 1.402 * Cr
    G = Y - 0.344136 * Cb - 0.714136 * Cr
    B = Y + 1.772 * Cb
    
    return np.clip(np.stack((B, G, R), axis=-1), 0, 255).astype(np.uint8)
