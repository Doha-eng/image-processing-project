import cv2
import numpy as np
from scipy import stats

class ImageProcessor:
     
    def addition(image, value):
        return cv2.add(image, np.full(image.shape, value, dtype=np.uint8))

     
    def subtraction(image, value):
        return cv2.subtract(image, np.full(image.shape, value, dtype=np.uint8))

     
    def division(image, value):
        if value == 0: return image
        return cv2.divide(image, np.full(image.shape, value, dtype=np.uint8))

     
    def complement(image):
        return cv2.bitwise_not(image)

     
    def change_lighting_color(image, red_val):
        img = image.copy()
        img[:, :, 2] = cv2.add(img[:, :, 2], red_val)
        return img

     
    def swap_channels_r_g(image):
        img = image.copy()
        r = img[:, :, 2].copy()
        g = img[:, :, 1].copy()
        img[:, :, 2] = g
        img[:, :, 1] = r
        return img

     
    def eliminate_red(image):
        img = image.copy()
        img[:, :, 2] = 0
        return img

     
    def histogram_stretching(image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        min_val = np.min(image)
        max_val = np.max(image)
        stretched = (image - min_val) * (255.0 / (max_val - min_val))
        return stretched.astype(np.uint8)

     
    def histogram_equalization(image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.equalizeHist(image)

     
    def average_filter(image, kernel_size=3):
        return cv2.blur(image, (kernel_size, kernel_size))

     
    def laplacian_filter(image):
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        return cv2.convertScaleAbs(laplacian)

     
    def max_filter(image, size=3):
        kernel = np.ones((size, size), np.uint8)
        return cv2.dilate(image, kernel)

     
    def min_filter(image, size=3):
        kernel = np.ones((size, size), np.uint8)
        return cv2.erode(image, kernel)

     
    def median_filter(image, size=3):
        return cv2.medianBlur(image, size)

     
    def mode_filter(image, size=3):
        
        def mode_func(window):
            return stats.mode(window, axis=None).mode
        
        return cv2.medianBlur(image, size) 

     
    def add_salt_and_pepper(image, amount=0.02):
        noisy = image.copy()
        # Salt
        num_salt = np.ceil(amount * image.size * 0.5)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        noisy[tuple(coords)] = 255
        # Pepper
        num_pepper = np.ceil(amount * image.size * 0.5)
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        noisy[tuple(coords)] = 0
        return noisy

     
    def outlier_method(image, threshold=50):
    
        kernel = np.ones((3,3), np.float32) / 9
        mean_img = cv2.filter2D(image, -1, kernel)
        diff = cv2.absdiff(image, mean_img)
        mask = diff > threshold
        result = image.copy()
        result[mask] = mean_img[mask]
        return result

     
    def add_gaussian_noise(image, mean=0, sigma=25):
        gauss = np.random.normal(mean, sigma, image.shape).astype('int16')
        noisy = image.astype('int16') + gauss
        return np.clip(noisy, 0, 255).astype('uint8')

     
    def thresholding_basic(image, thresh=127):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, res = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)
        return res

     
    def thresholding_automatic(image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, res = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return res

     
    def thresholding_adaptive(image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

     
    def sobel_detector(image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        return cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

     
    def dilation(image, size=3):
        kernel = np.ones((size, size), np.uint8)
        return cv2.dilate(image, kernel, iterations=1)

     
    def erosion(image, size=3):
        kernel = np.ones((size, size), np.uint8)
        return cv2.erode(image, kernel, iterations=1)

     
    def opening(image, size=3):
        kernel = np.ones((size, size), np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

     
    def internal_boundary(image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((3,3), np.uint8)
        eroded = cv2.erode(image, kernel, iterations=1)
        return cv2.subtract(image, eroded)

     
    def external_boundary(image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(image, kernel, iterations=1)
        return cv2.subtract(dilated, image)

     
    def morphological_gradient(image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((3,3), np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
