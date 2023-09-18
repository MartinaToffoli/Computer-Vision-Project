import cv2
import numpy as np

def image_processing(image):
    im = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(im, (28, 28), interpolation=cv2.INTER_AREA)
    lighter = cv2.add(resized, np.array([40.0]))
    bilateral = cv2.bilateralFilter(lighter, 9, 75, 75)
    return bilateral
