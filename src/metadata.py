import cv2
import numpy as np
import struct
import pickle

def save_metadata(original_height: int, original_width: int, filename: str, scaled_quant_matrix_Y: np.ndarray, scaled_quant_matrix_C: np.ndarray, huffman_codes: dict, Y_encoded: str, Cb_encoded: str, Cr_encoded: str) -> None:
    """
    Guarda los metadatos de la imagen comprimida en un archivo.
    
    Args:
        original_height (int): Altura original de la imagen.
        original_width (int): Ancho original de la imagen.
        filename (str): Nombre del archivo comprimido.
        scaled_quant_matrix_Y (np.ndarray): Matriz de cuantización escalada para el canal Y.
        scaled_quant_matrix_C (np.ndarray): Matriz de cuantización escalada para los canales Cb y Cr.
        huffman_codes (dict): Códigos de Huffman.
        Y_encoded (str): Datos codificados del canal Y.
        Cb_encoded (str): Datos codificados del canal Cb.
        Cr_encoded (str): Datos codificados del canal Cr.
    """

    jpeg_header = b'\xFF\xD8'         # Inicio de la imagen JPEG

    # SOF0 (Start of Frame 0) Header
    sof0_header = struct.pack(
        ">BBHHBBBB",
        0xFF,                         # Marca JPEG (Start of Frame)
        0xC0,                         # Tipo de segmento (Baseline DCT)
        original_height,
        original_width,
        0x03,                         # Número de componentes
        0x11,                         # Submuestreo
        0x22,                         # Submuestreo
        0x22,                         # Submuestreo
    )
    
    # DQT (Define Quantization Table) Header
    dqt_header = struct.pack(
        ">BBB",
        0xFF,                         # Marca JPEG (Define Quantization Table)
        0xDB,                         # Tipo de segmento (Quantization Table)
        0x00,                         # Identificador de tabla de cuantización (0 = Y)
    ) + scaled_quant_matrix_Y.flatten().tobytes() + struct.pack(
        ">BBB",
        0xFF,                         # Marca JPEG (Define Quantization Table)
        0xDB,                         # Tipo de segmento (Quantization Table)
        0x01,                         # Identificador de tabla de cuantización (1 = C)
    ) + scaled_quant_matrix_C.flatten().tobytes()
        
    # DHT (Define Huffman Table) Header
    huffman_header = pickle.dumps(huffman_codes)
    dht_header = struct.pack(
        ">BB",
        0xFF,                         # Marca JPEG (Define Huffman Table)
        0xC4,                         # Tipo de segmento (Huffman Table)
    ) + huffman_header
    
    # SOS (Start of Scan) Header
    Y_encoded_length = struct.pack('>I', len(Y_encoded))
    Y_encoded_header = int(Y_encoded, 2).to_bytes((len(Y_encoded) + 7) // 8, byteorder='big')
    
    Cb_encoded_length = struct.pack('>I', len(Cb_encoded))
    Cb_encoded_header = int(Cb_encoded, 2).to_bytes((len(Cb_encoded) + 7) // 8, byteorder='big')
    
    Cr_encoded_length = struct.pack('>I', len(Cr_encoded))
    Cr_encoded_header = int(Cr_encoded, 2).to_bytes((len(Cr_encoded) + 7) // 8, byteorder='big')
    
    encoded_header = Y_encoded_header + Cb_encoded_header + Cr_encoded_header
    
    sos_header = struct.pack(
        ">BB",
        0xFF,                         # Marca JPEG (Start of Scan)
        0xDA,                         # Tipo de segmento (Start of Scan)
    ) + Y_encoded_length + Cb_encoded_length + Cr_encoded_length + encoded_header
    
    with open(filename, "wb") as file:
        file.write(jpeg_header)
        file.write(sof0_header)
        file.write(dqt_header)
        file.write(dht_header)
        file.write(sos_header)

def load_metadata(filename: str) -> tuple:
    """
    Carga los metadatos de la imagen comprimida desde un archivo.
    
    Args:
        filename (str): Nombre del archivo comprimido.
        
    Returns:
        tuple: Un tuple con los metadatos de la imagen comprimida.
    """
    with open(filename, "rb") as file:
        jpeg_data = file.read()

    jpeg_header = jpeg_data[:2]
    if jpeg_header != b'\xFF\xD8':
        raise ValueError("El archivo no es una imagen JPEG válida.")
    
    original_height, original_width = struct.unpack(">HH", jpeg_data[4:8])
    
    scaled_quant_matrix_Y = np.frombuffer(jpeg_data[15:79], dtype=np.uint8).reshape(8, 8)
    scaled_quant_matrix_C = np.frombuffer(jpeg_data[82:146], dtype=np.uint8).reshape(8, 8)
    
    encoded_header_start = jpeg_data.find(b'\xFF\xDA')
    huffman_codes = pickle.loads(jpeg_data[148:encoded_header_start])

    Y_encoded_length = struct.unpack(">I", jpeg_data[encoded_header_start + 2:encoded_header_start + 6])[0]
    Cb_encoded_length = struct.unpack(">I", jpeg_data[encoded_header_start + 6:encoded_header_start + 10])[0]
    Cr_encoded_length = struct.unpack(">I", jpeg_data[encoded_header_start + 10:encoded_header_start + 14])[0]
    
    Y_encoded = bin(int.from_bytes(jpeg_data[encoded_header_start + 14:encoded_header_start + 14 + (Y_encoded_length + 7) // 8], byteorder='big'))[2:].zfill(Y_encoded_length)
    Cb_encoded = bin(int.from_bytes(jpeg_data[encoded_header_start + 14 + (Y_encoded_length + 7) // 8:encoded_header_start + 14 + (Y_encoded_length + 7) // 8 + (Cb_encoded_length + 7) // 8], byteorder='big'))[2:].zfill(Cb_encoded_length)
    Cr_encoded = bin(int.from_bytes(jpeg_data[encoded_header_start + 14 + (Y_encoded_length + 7) // 8 + (Cb_encoded_length + 7) // 8:], byteorder='big'))[2:].zfill(Cr_encoded_length)  
    
    return original_height, original_width, scaled_quant_matrix_Y, scaled_quant_matrix_C, huffman_codes, Y_encoded, Cb_encoded, Cr_encoded
