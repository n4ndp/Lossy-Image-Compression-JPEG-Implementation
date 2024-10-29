import math
import numpy as np

def zigzag(matrix: np.ndarray) -> np.ndarray:
    height, width = matrix.shape
    zigzag_order_matrix = np.empty(height * width, dtype=matrix.dtype)
    index = 0
    
    for s in range(height + width - 1):
        if s % 2 == 0:
            for i in range(max(0, s - width + 1), min(s + 1, height)):
                zigzag_order_matrix[index] = matrix[i, s - i]
                index += 1
        else:
            for i in range(max(0, s - height + 1), min(s + 1, width)):
                zigzag_order_matrix[index] = matrix[s - i, i]
                index += 1

    return zigzag_order_matrix

def jpeg_encoding():
    pass