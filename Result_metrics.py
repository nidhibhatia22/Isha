import cv2
import numpy
import numpy as np
from PIL import Image, ImageChops
import math

class SystemEvaluation:


    def calculate_psnr(self, img1, img2, max_value=255):
        """"Calculating peak signal-to-noise ratio (PSNR) between two images."""
        mse = np.mean((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)
        if mse == 0:
            return 100
        return 20 * np.log10(max_value / (np.sqrt(mse)))

    def mse(self, img1, img2):
       h, w,c = img1.shape
       diff = cv2.subtract(img1, img2)
       err = np.sum(diff**2)
       mse = err/(float(h*w))
       mse=mse/10
       return mse

    def ssim(self, image1, image2):
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        img1 = np.array(image1, dtype=np.float32)
        img2 = np.array(image2, dtype=np.float32)

        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                (sigma1_sq + sigma2_sq + C2))
        ssim_map = ssim_map.mean()
        return ssim_map