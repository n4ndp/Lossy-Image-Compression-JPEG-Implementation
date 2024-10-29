import math
import numpy as np

def matrix_C(n: int) -> np.ndarray:
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
