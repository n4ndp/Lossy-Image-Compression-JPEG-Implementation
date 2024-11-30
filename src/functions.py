import numpy as np

def apply_padding(image: np.ndarray) -> np.ndarray:
    """
    Aplica relleno a una imagen para que sus dimensiones sean múltiplos de 8.
    
    Args:
        image (np.ndarray): Imagen a la que se le aplicará el relleno.
        
    Returns:
        np.ndarray: Imagen con relleno aplicado.
    """
    if image is None or image.size == 0:
        raise ValueError("La imagen de entrada está vacía.")
    
    height, width = image.shape
    height_padding = (8 - height % 8) % 8
    width_padding = (8 - width % 8) % 8
    
    return np.pad(image, ((0, height_padding), (0, width_padding)), mode="constant")

def remove_padding(image: np.ndarray, original_height: int, original_width: int) -> np.ndarray:
    """
    Elimina el relleno aplicado a una imagen.
    
    Args:
        image (np.ndarray): Imagen a la que se le eliminará el relleno.
        original_height (int): Altura original de la imagen.
        original_width (int): Ancho original de la imagen.
        
    Returns:
        np.ndarray: Imagen sin relleno aplicado.
    """
    return image[:original_height, :original_width]

def downsample(ycbcr_image: np.ndarray) -> tuple:
    """
    Realiza el muestreo de color en una imagen en formato YCbCr.
    
    Args:
        ycbcr_image (np.ndarray): Imagen en formato YCbCr.
        
    Returns:
        tuple: Tupla con los canales Y, Cb y Cr muestreados.
    """
    Y = ycbcr_image[:, :, 0]
    Cb_downsampled = ycbcr_image[:, :, 1][::2, ::2]
    Cr_downsampled = ycbcr_image[:, :, 2][::2, ::2]
    
    return Y, Cb_downsampled, Cr_downsampled

def upsample(Y: np.ndarray, Cb_downsampled: np.ndarray, Cr_downsampled: np.ndarray) -> np.ndarray:
    """
    Realiza el muestreo de color en una imagen en formato YCbCr.
    
    Args:
        Y (np.ndarray): Canal Y de la imagen.
        Cb_downsampled (np.ndarray): Canal Cb muestreado.
        Cr_downsampled (np.ndarray): Canal Cr muestreado.
        
    Returns:
        np.ndarray: Imagen en formato YCbCr con el muestreo de color aplicado.
    """
    Cr = np.repeat(np.repeat(Cr_downsampled, 2, axis=0), 2, axis=1)[:Y.shape[0], :Y.shape[1]]
    Cb = np.repeat(np.repeat(Cb_downsampled, 2, axis=0), 2, axis=1)[:Y.shape[0], :Y.shape[1]]
    
    return np.stack((Y, Cb, Cr), axis=-1)
