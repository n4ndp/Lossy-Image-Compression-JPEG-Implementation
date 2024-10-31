import cv2
from src.bgr_ycrcb_conversion import bgr_image_to_ycrcb, ycrcb_image_to_bgr
from src.functions import pad, unpad, downsampling, upsampling
from src.discrete_cosine_transform import discrete_cosine_transform
from src.quantization import quant_matrix_C, quant_matrix_Y, scale_quant_matrix, quantize
# from src.jpeg_encoding import jpeg_encoding

if __name__ == "__main__":
    image = cv2.imread('img/lena.bmp')
    
    ycrcb_image = bgr_image_to_ycrcb(image)
    
    Y, Cr_downsampled, Cb_downsampled = downsampling(ycrcb_image)
    
    Y_padded = pad(Y)
    Cr_downsampled_padded = pad(Cr_downsampled)
    Cb_downsampled_padded = pad(Cb_downsampled)
    
    Y_transformed = discrete_cosine_transform(Y_padded)
    Cr_transformed = discrete_cosine_transform(Cr_downsampled_padded)
    Cb_transformed = discrete_cosine_transform(Cb_downsampled_padded)
    
    quality = 50
    scaled_quant_matrix_Y = scale_quant_matrix(quant_matrix_Y, quality)
    scaled_quant_matrix_C = scale_quant_matrix(quant_matrix_C, quality)

    Y_quantized = quantize(Y_transformed, scaled_quant_matrix_Y)
    Cr_quantized = quantize(Cr_transformed, scaled_quant_matrix_C)
    Cb_quantized = quantize(Cb_transformed, scaled_quant_matrix_C)

    Y_dequantized = quantize(Y_quantized, scaled_quant_matrix_Y, inverse=True)
    Cr_dequantized = quantize(Cr_quantized, scaled_quant_matrix_C, inverse=True)
    Cb_dequantized = quantize(Cb_quantized, scaled_quant_matrix_C, inverse=True)

    Y_restored = discrete_cosine_transform(Y_dequantized, inverse=True)
    Cr_restored = discrete_cosine_transform(Cr_dequantized, inverse=True)
    Cb_restored = discrete_cosine_transform(Cb_dequantized, inverse=True)
    
    Y_restored = unpad(Y_restored, Y.shape[0], Y.shape[1])
    Cr_restored = unpad(Cr_restored, Cr_downsampled.shape[0], Cr_downsampled.shape[1])
    Cb_restored = unpad(Cb_restored, Cb_downsampled.shape[0], Cb_downsampled.shape[1])
    
    ycrcb_image_restored = upsampling(Y_restored, Cr_restored, Cb_restored)
    
    bgr_image_restored = ycrcb_image_to_bgr(ycrcb_image_restored)
    
    cv2.imshow('Original Image', image)
    cv2.imshow('Restored Image', bgr_image_restored)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
