import cv2
import numpy as np

def bgr_image_to_ycrcb(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float64)
    ycrcb_image = np.zeros_like(image, dtype=np.float64)

    B = image[:, :, 0]
    G = image[:, :, 1]
    R = image[:, :, 2]

    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 128
    Cb = (B - Y) * 0.564 + 128

    ycrcb_image[:, :, 0] = Y
    ycrcb_image[:, :, 1] = Cr
    ycrcb_image[:, :, 2] = Cb

    return np.round(ycrcb_image).astype(np.uint8)

def ycrcb_image_to_bgr(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float64)
    bgr_image = np.zeros_like(image, dtype=np.float64)

    Y = image[:, :, 0]
    Cr = image[:, :, 1] - 128
    Cb = image[:, :, 2] - 128

    R = Y + 1.403 * Cr
    G = Y - 0.344 * Cb - 0.714 * Cr
    B = Y + 1.770 * Cb

    bgr_image[:, :, 0] = B
    bgr_image[:, :, 1] = G
    bgr_image[:, :, 2] = R

    bgr_image[bgr_image > 255] = 255
    bgr_image[bgr_image < 0] = 0

    return np.round(bgr_image).astype(np.uint8)

if __name__ == "__main__":
    image = cv2.imread('img/frutas.bmp')
    
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
