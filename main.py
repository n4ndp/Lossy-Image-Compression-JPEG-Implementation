import cv2
import numpy as np
import argparse
from datetime import datetime

from src.bgr_ycrcb_conversion import bgr_image_to_ycbcr, ycbcr_image_to_bgr
from src.functions import apply_padding, remove_padding, downsample, upsample
from src.discrete_cosine_transform import apply_discrete_cosine_transform
from src.quantization import QTY, QTC, scale_quant_matrix, quantize
from src.run_length_encoding import zigzag, run_length_encode, undo_zigzag
from src.huffman_encoding import _encoded_data, huffman_encode, huffman_decode
from src.metadata import save_metadata, load_metadata

def compress_image(image: np.ndarray, quality: int = 50, filename: str = "compressed_image.jpeg") -> datetime:
    """
    Comprime una imagen utilizando el algoritmo JPEG.
    
    Args:
        image (np.ndarray): Imagen a comprimir.
        quality (int, opcional): Calidad de la compresión. Por defecto es 50.
        filename (str, opcional): Nombre del archivo comprimido. Por defecto es "compressed_image.jpeg".
    """
    start = datetime.now()
    
    ycbcr_image = bgr_image_to_ycbcr(image)
    
    Y, Cb, Cr = downsample(ycbcr_image)
    
    Y = apply_padding(Y)
    Cb = apply_padding(Cb)
    Cr = apply_padding(Cr)
    
    Y = apply_discrete_cosine_transform(Y)
    Cb = apply_discrete_cosine_transform(Cb)
    Cr = apply_discrete_cosine_transform(Cr)
    
    scaled_quant_matrix_Y = scale_quant_matrix(QTY, quality)
    scaled_quant_matrix_C = scale_quant_matrix(QTC, quality)
    
    Y = quantize(Y, scaled_quant_matrix_Y)
    Cb = quantize(Cb, scaled_quant_matrix_C)
    Cr = quantize(Cr, scaled_quant_matrix_C)
    
    Y_zigzag_rle = [run_length_encode(zigzag(Y[i:i+8, j:j+8])) for i in range(0, Y.shape[0], 8) for j in range(0, Y.shape[1], 8)]
    Cb_zigzag_rle = [run_length_encode(zigzag(Cb[i:i+8, j:j+8])) for i in range(0, Cb.shape[0], 8) for j in range(0, Cb.shape[1], 8)]
    Cr_zigzag_rle = [run_length_encode(zigzag(Cr[i:i+8, j:j+8])) for i in range(0, Cr.shape[0], 8) for j in range(0, Cr.shape[1], 8)]
        
    Y_flat_rle = [item for sublist in Y_zigzag_rle for item in sublist]
    Cb_flat_rle = [item for sublist in Cb_zigzag_rle for item in sublist]
    Cr_flat_rle = [item for sublist in Cr_zigzag_rle for item in sublist]

    Y_flat_rle = [item for sublist in Y_flat_rle for item in sublist]
    Cb_flat_rle = [item for sublist in Cb_flat_rle for item in sublist]
    Cr_flat_rle = [item for sublist in Cr_flat_rle for item in sublist]
    
    all_flat_rle = Y_flat_rle + Cb_flat_rle + Cr_flat_rle
    huffman_codes, _ = huffman_encode(all_flat_rle)
    
    original_height, original_width = image.shape[:2]
    Y_encoded = _encoded_data(Y_flat_rle, huffman_codes)
    Cb_encoded = _encoded_data(Cb_flat_rle, huffman_codes)
    Cr_encoded = _encoded_data(Cr_flat_rle, huffman_codes)
    
    save_metadata(original_height, original_width, filename, scaled_quant_matrix_Y, scaled_quant_matrix_C, huffman_codes, Y_encoded, Cb_encoded, Cr_encoded)
    
    return datetime.now() - start

def decompress_image(filename: str) -> tuple:
    """
    Descomprime una imagen comprimida utilizando el algoritmo JPEG.
    
    Args:
        filename (str): Nombre del archivo comprimido.
        
    Returns:
        np.ndarray: Imagen descomprimida.
    """
    start = datetime.now()
    
    original_height, original_width, scaled_quant_matrix_Y, scaled_quant_matrix_C, huffman_codes, Y_encoded, Cb_encoded, Cr_encoded = load_metadata(filename)
    
    Y_flat_rle = huffman_decode(Y_encoded, huffman_codes)
    Cb_flat_rle = huffman_decode(Cb_encoded, huffman_codes)
    Cr_flat_rle = huffman_decode(Cr_encoded, huffman_codes)
    
    def _run_length_decode(rle_data):
        blocks = []
        block = []
        count = 0
        for freq, val in rle_data:
            block.extend([val] * freq)
            count += freq
            while count >= 64:
                blocks.append(block[:64])
                block = block[64:]
                count -= 64
        return blocks

    Y_flat_rle_ = _run_length_decode([(Y_flat_rle[i], Y_flat_rle[i+1]) for i in range(0, len(Y_flat_rle), 2)])
    Cb_flat_rle_ = _run_length_decode([(Cb_flat_rle[i], Cb_flat_rle[i+1]) for i in range(0, len(Cb_flat_rle), 2)])
    Cr_flat_rle_ = _run_length_decode([(Cr_flat_rle[i], Cr_flat_rle[i+1]) for i in range(0, len(Cr_flat_rle), 2)])

    Y_height = (original_height + 7) // 8 * 8
    Y_width = (original_width + 7) // 8 * 8
    C_height = ((original_height + 1) // 2 + 7) // 8 * 8
    C_width = ((original_width + 1) // 2 + 7) // 8 * 8
    
    Y = np.zeros((Y_height, Y_width), dtype=np.float32)
    Cb = np.zeros((C_height, C_width), dtype=np.float32)
    Cr = np.zeros((C_height, C_width), dtype=np.float32)
    
    def _fill_blocks(image_matrix, blocks):
        for i in range(0, image_matrix.shape[0], 8):
            for j in range(0, image_matrix.shape[1], 8):
                block = np.array(blocks.pop(0))
                image_matrix[i:i+8, j:j+8] = undo_zigzag(block, 8)

    _fill_blocks(Y, Y_flat_rle_)
    _fill_blocks(Cb, Cb_flat_rle_)
    _fill_blocks(Cr, Cr_flat_rle_)
    
    Y = quantize(Y, scaled_quant_matrix_Y, inverse=True)
    Cb = quantize(Cb, scaled_quant_matrix_C, inverse=True)
    Cr = quantize(Cr, scaled_quant_matrix_C, inverse=True)
    
    Y = apply_discrete_cosine_transform(Y, inverse=True)
    Cb = apply_discrete_cosine_transform(Cb, inverse=True)
    Cr = apply_discrete_cosine_transform(Cr, inverse=True)
    
    Y = remove_padding(Y, original_height, original_width)
    Cb = remove_padding(Cb, original_height // 2, original_width // 2)
    Cr = remove_padding(Cr, original_height // 2, original_width // 2)
    
    ycbcr_image = upsample(Y, Cb, Cr)
    bgr_image = ycbcr_image_to_bgr(ycbcr_image)
    
    return bgr_image, datetime.now() - start

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compresión y descompresión de imágenes JPEG.")
    parser.add_argument("image_path", type=str, help="Ruta de la imagen a comprimir/descomprimir.")
    parser.add_argument("quality", type=int, help="Calidad de la compresión (1-100).")
    parser.add_argument("output_file", type=str, help="Nombre del archivo de salida para la imagen comprimida.")
    args = parser.parse_args()
    
    image = cv2.imread(args.image_path)

    compression_time = compress_image(image, args.quality, args.output_file)
    decompressed_image, decompression_time = decompress_image(args.output_file)
    
    print(f"Compression time: {compression_time}")
    print(f"Decompression time: {decompression_time}")
    
    cv2.imshow("Original Image", image)
    cv2.imshow("Decompressed Image", decompressed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
