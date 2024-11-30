import numpy as np

def zigzag(matrix: np.ndarray) -> np.ndarray:
    """
    Recorre una matriz en orden zigzag.
    
    Args:
        matrix (np.ndarray): Matriz a recorrer.
        
    Returns:
        np.ndarray: Matriz recorrida en orden zigzag.
    """
    if matrix is None or matrix.size == 0:
        raise ValueError("La matriz de entrada está vacía.")
    
    n = matrix.shape[0]
    zigzag = np.zeros(n * n, dtype=np.int16)
    row, col = 0, 0
    index = 0
    going_up = True
    
    while row < n and col < n:
        zigzag[index] = matrix[row, col]
        index += 1
        
        if going_up:
            if row == 0 and col < n - 1:
                col += 1
                going_up = False
            elif col == n - 1:
                row += 1
                going_up = False
            else:
                row -= 1
                col += 1
        else:
            if col == 0 and row < n - 1:
                row += 1
                going_up = True
            elif row == n - 1:
                col += 1
                going_up = True
            else:
                row += 1
                col -= 1
                
    return zigzag

def undo_zigzag(zigzag: np.ndarray, n: int) -> np.ndarray:
    """
    Recorre un arreglo en orden zigzag y lo convierte en una matriz.
    
    Args:
        zigzag (np.ndarray): Arreglo en orden zigzag.
        n (int): Tamaño de la matriz.
        
    Returns:
        np.ndarray: Matriz generada a partir del arreglo zigzag.
    """
    if zigzag is None or zigzag.size == 0:
        raise ValueError("El arreglo de entrada está vacío.")
    
    if n <= 0:
        raise ValueError("El tamaño de la matriz debe ser mayor a cero.")
    
    matrix = np.zeros((n, n), dtype=np.int16)
    row, col = 0, 0
    index = 0
    going_up = True
    
    while row < n and col < n:
        matrix[row, col] = zigzag[index]
        index += 1
        
        if going_up:
            if row == 0 and col < n - 1:
                col += 1
                going_up = False
            elif col == n - 1:
                row += 1
                going_up = False
            else:
                row -= 1
                col += 1
        else:
            if col == 0 and row < n - 1:
                row += 1
                going_up = True
            elif row == n - 1:
                col += 1
                going_up = True
            else:
                row += 1
                col -= 1
                
    return matrix

def run_length_encode(zigzag: np.ndarray) -> list:
    """
    Codifica un arreglo en orden zigzag utilizando el algoritmo Run Length Encoding.
    
    Args:
        zigzag (np.ndarray): Arreglo en orden zigzag.
        
    Returns:
        list: Lista con los elementos codificados.
    """
    if zigzag is None or zigzag.size == 0:
        raise ValueError("El arreglo de entrada está vacío.")
    
    encoded = []
    count = 0
    current = zigzag[0]
    
    for value in zigzag:
        if value == current:
            count += 1
        else:
            encoded.append((count, current))
            count = 1
            current = value
            
    encoded.append((count, current))
    
    return encoded
