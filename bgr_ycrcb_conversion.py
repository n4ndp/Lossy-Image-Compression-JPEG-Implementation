import cv2
import numpy as np

def bgr_image_to_ycrcb(image: np.ndarray) -> np.ndarray:
    """Convierte una imagen en formato BGR a YCrCb.
    
    Args:
        image (np.ndarray): Imagen en formato BGR.
        
    Returns:
        np.ndarray: Imagen convertida en formato YCrCb con valores en uint8.
    """
    image = image.astype(np.float32)

    B = image[:, :, 0]
    G = image[:, :, 1]
    R = image[:, :, 2]

    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 128
    Cb = (B - Y) * 0.564 + 128

    ycrcb_image = np.stack((Y, Cr, Cb), axis=-1)

    return np.round(ycrcb_image).astype(np.uint8)

def ycrcb_image_to_bgr(image: np.ndarray) -> np.ndarray:
    """Convierte una imagen en formato YCrCb a BGR.

    Args:
        image (np.ndarray): Imagen en formato YCrCb.
        
    Returns:
        np.ndarray: Imagen convertida en formato BGR con valores en uint8.
    """
    image = image.astype(np.float32)

    Y = image[:, :, 0]
    Cr = image[:, :, 1] - 128
    Cb = image[:, :, 2] - 128

    R = Y + 1.403 * Cr
    G = Y - 0.344 * Cb - 0.714 * Cr
    B = Y + 1.770 * Cb

    bgr_image = np.stack((B, G, R), axis=-1)
    bgr_image = np.clip(bgr_image, 0, 255)

    return np.round(bgr_image).astype(np.uint8)

if __name__ == "__main__":
    image = cv2.imread('img/lena.bmp')
    
    my_ycrcb = bgr_image_to_ycrcb(image)
    cv_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    
    print("My YCrCb:")
    print(my_ycrcb)
    print("OpenCV YCrCb:")
    print(cv_ycrcb)
    
    my_bgr = ycrcb_image_to_bgr(my_ycrcb)
    cv_bgr = cv2.cvtColor(cv_ycrcb, cv2.COLOR_YCrCb2BGR)
    
    print("My BGR:")
    print(my_bgr)
    print("OpenCV BGR:")
    print(cv_bgr)
    
    cv2.imshow('Original', image)
    cv2.imshow('My BGR', my_bgr)
    cv2.imshow('OpenCV BGR', cv_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
