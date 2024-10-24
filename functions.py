import math
import numpy as np

def pad(channel: np.ndarray) -> np.ndarray:
    """Añade padding a un canal para que sus dimensiones sean múltiplos de 8.
    
    Este padding se realiza con ceros y se añade a la derecha y abajo del canal. 
    Si las dimensiones del canal ya son múltiplos de 8, no se añade padding.

    Args:
        channel (np.ndarray): El canal de imagen (Y, Cr, o Cb).

    Returns:
        np.ndarray: El canal padded con dimensiones que son múltiplos de 8, 
        o el canal original si ya cumple con esta condición.
    """
    height, width = channel.shape
    
    if height % 8 == 0 and width % 8 == 0:
        return channel

    height_padding = (8 - height % 8) % 8
    width_padding = (8 - width % 8) % 8

    padded_channel = np.pad(channel, ((0, height_padding), (0, width_padding)), mode='constant')

    return padded_channel

def unpad(channel: np.ndarray, original_height: int, original_width: int) -> np.ndarray:
    """Elimina el padding de un canal, restaurando sus dimensiones originales.

    Si el canal ya tiene las dimensiones originales, no se realiza ningún cambio.
    
    Args:
        channel (np.ndarray): El canal con padding (Y, Cr o Cb).
        original_height (int): La altura original del canal antes de añadir el padding.
        original_width (int): El ancho original del canal antes de añadir el padding.

    Returns:
        np.ndarray: El canal sin padding, con sus dimensiones originales.
    """
    height, width = channel.shape
    
    if height == original_height and width == original_width:
        return channel
    
    unpadded_channel = channel[:original_height, :original_width]
    
    return unpadded_channel

def downsampling(ycrcb_image: np.ndarray) -> tuple:
    """Aplica subsampling 4:2:0 a los canales Cr y Cb.

    Reduce las dimensiones de los canales Cr y Cb en un factor de 2 en ambas
    direcciones, mientras mantiene el canal Y intacto.

    Args:
        ycrcb_image (np.ndarray): Imagen en formato YCrCb.

    Returns:
        tuple: Contiene:
            - Y (np.ndarray): Canal de luminancia (Y) sin subsampling.
            - Cr_downsampled (np.ndarray): Canal Cr con subsampling.
            - Cb_downsampled (np.ndarray): Canal Cb con subsampling.
    """
    Y = ycrcb_image[:, :, 0]
    Cr_downsampled = ycrcb_image[:, :, 1][::2, ::2]
    Cb_downsampled = ycrcb_image[:, :, 2][::2, ::2]

    return Y, Cr_downsampled, Cb_downsampled

def upsampling(Y: np.ndarray, Cr_downsampled: np.ndarray, Cb_downsampled: np.ndarray) -> np.ndarray:
    """Aplica upsampling a los canales Cr y Cb para restaurar su tamaño original.

    Los canales Cr y Cb se expanden duplicando filas y columnas para coincidir
    con las dimensiones del canal Y.

    Args:
        Y (np.ndarray): Canal de luminancia (Y) con resolución completa.
        Cr_downsampled (np.ndarray): Canal Cr con subsampling.
        Cb_downsampled (np.ndarray): Canal Cb con subsampling.

    Returns:
        np.ndarray: Imagen en formato YCrCb con los canales restaurados.
    """
    Cr = np.repeat(np.repeat(Cr_downsampled, 2, axis=0), 2, axis=1)
    Cb = np.repeat(np.repeat(Cb_downsampled, 2, axis=0), 2, axis=1)
    
    Cr = Cr[:Y.shape[0], :Y.shape[1]]
    Cb = Cb[:Y.shape[0], :Y.shape[1]]
    
    ycrcb_image = np.stack((Y, Cr, Cb), axis=-1)

    return ycrcb_image
