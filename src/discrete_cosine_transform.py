import numpy as np

def matrix_C(n: int) -> np.ndarray:
    """
    Genera la matriz C para la transformada discreta del coseno.
    
    Args:
        n (int): Tamaño de la matriz.
        
    Returns:
        np.ndarray: Matriz C para la transformada discreta del coseno.
    """
    C = np.zeros((n, n), dtype=np.float32)
    for u in range(n):
        for v in range(n):
            if u == 0:
                C[u, v] = 1 / np.sqrt(n)
            else:
                C[u, v] = np.sqrt(2 / n) * np.cos(((2 * v + 1) * u * np.pi) / (2 * n))
    
    return C

def apply_discrete_cosine_transform(channel: np.ndarray, inverse: bool = False) -> np.ndarray:
    """
    Aplica la transformada discreta del coseno a un canal de la imagen.
    
    Args:
        channel (np.ndarray): Canal de la imagen a transformar.
        inverse (bool, opcional): Si es `True`, se aplica la transformada inversa, 
                                 si es `False`, se aplica la transformada normal. Por defecto es `False`.
                                 
    Returns:
        np.ndarray: Canal transformado.
    """
    if channel is None or channel.size == 0:
        raise ValueError("El canal de entrada está vacío.")
    
    if channel.shape[0] % 8 != 0 or channel.shape[1] % 8 != 0:
        raise ValueError("Las dimensiones del canal deben ser múltiplos de 8.")
    
    n = 8
    height, width = channel.shape
    C = matrix_C(n)
    C_T = C.T
    transformed_channel = np.zeros_like(channel, dtype=np.float32)
    
    for y in range(0, height, n):
        for x in range(0, width, n):
            block = channel[y:y+n, x:x+n]
            
            if inverse:
                transformed_channel[y:y+n, x:x+n] = np.matmul(np.matmul(C_T, block), C)
            else:
                transformed_channel[y:y+n, x:x+n] = np.matmul(np.matmul(C, block), C_T)
                
    return transformed_channel.astype(np.float32)
