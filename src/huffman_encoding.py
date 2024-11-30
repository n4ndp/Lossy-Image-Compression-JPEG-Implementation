import heapq
from collections import defaultdict

class HuffmanNode:
    def __init__(self, value, frequency):
        self.value = value
        self.frequency = frequency
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.frequency < other.frequency

def build_huffman_tree(frequency_map: defaultdict) -> HuffmanNode:
    """
    Construye un árbol de Huffman a partir de un diccionario de frecuencias.
    
    Args:
        frequency_map (defaultdict): Diccionario de frecuencias.
        
    Returns:
        HuffmanNode: Raíz del árbol de Huffman.
    """
    priority_queue = [HuffmanNode(value, frequency) for value, frequency in frequency_map.items()]
    heapq.heapify(priority_queue)

    while len(priority_queue) > 1:
        left = heapq.heappop(priority_queue)
        right = heapq.heappop(priority_queue)
        
        internal_node = HuffmanNode(None, left.frequency + right.frequency)
        internal_node.left = left
        internal_node.right = right
        
        heapq.heappush(priority_queue, internal_node)
    
    return priority_queue[0]

def generate_huffman_codes(node: HuffmanNode, prefix: str = "") -> dict:
    """
    Genera los códigos de Huffman a partir de un árbol de Huffman.
    
    Args:
        node (HuffmanNode): Nodo raíz del árbol de Huffman.
        prefix (str, opcional): Prefijo a utilizar. Por defecto es una cadena vacía.
        
    Returns:
        dict: Diccionario con los códigos de Huffman.
    """
    if node is None:
        return {}
    
    if node.value is not None:
        return {node.value: prefix}
    
    codes = {}
    codes.update(generate_huffman_codes(node.left, prefix + "0"))
    codes.update(generate_huffman_codes(node.right, prefix + "1"))
    
    return codes

def _encoded_data(flat_rle_sequence: list, huffman_codes: dict) -> str:
    """
    Codifica una secuencia plana utilizando los códigos de Huffman.
    
    Args:
        flat_rle_sequence (list): Secuencia plana a codificar.
        huffman_codes (dict): Diccionario con los códigos de Huffman.
        
    Returns:
        str: Secuencia codificada.
    """
    return ''.join(huffman_codes[value] for value in flat_rle_sequence)

def huffman_encode(flat_rle_sequence: list) -> tuple:
    """
    Codifica una secuencia plana utilizando el algoritmo de Huffman
    
    Args:
        flat_rle_sequence (list): Secuencia plana a codificar.
        
    Returns:
        tuple: Un tuple con los códigos de Huffman y los datos codificados.
    """    
    frequency_map = defaultdict(int)
    for value in flat_rle_sequence:
        frequency_map[value] += 1

    root = build_huffman_tree(frequency_map)
    huffman_codes = generate_huffman_codes(root)

    return huffman_codes, _encoded_data(flat_rle_sequence, huffman_codes)

def huffman_decode(encoded_data: str, huffman_codes: dict) -> list:
    """
    Decodifica una secuencia codificada utilizando el algoritmo de Huffman.
    
    Args:
        encoded_data (str): Secuencia codificada.
        huffman_codes (dict): Diccionario con los códigos de Huffman.
        
    Returns:
        list: Secuencia decodificada.
    """
    reversed_codes = {v: k for k, v in huffman_codes.items()}
    
    current_code = ""
    decoded_data = []
    
    for bit in encoded_data:
        current_code += bit
        if current_code in reversed_codes:
            decoded_data.append(reversed_codes[current_code])
            current_code = ""
    
    return decoded_data
