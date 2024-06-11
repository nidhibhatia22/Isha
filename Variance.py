import cv2
import numpy as np
from scipy import ndimage

img = cv2.imread("..//Image_output//Noise_removed.png")
lbl, nlbl = ndimage.label(img)
ndimage.variance(img, lbl, index=np.arange(1, nlbl+1))
var = ndimage.variance(img, lbl)
print(var/100)