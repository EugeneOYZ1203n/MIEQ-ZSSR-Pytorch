from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np

def basic_denoise_sharpen_contrast():
    def processing(img):
        img_denoised = img.filter(ImageFilter.GaussianBlur(radius=1))
        img_sharpened = img_denoised.filter(ImageFilter.SHARPEN)

        enhancer = ImageEnhance.Contrast(img_sharpened)
        img_contrast = enhancer.enhance(1.2)

        return img_contrast
    return processing

def denoise_gaussianBlur(radius=2):
    def processing(img):
        img_denoised = img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img_denoised
    return processing

def denoise_fastN1MeansDenoising(templateWindowSize=7, searchWindowSize=21):
    def processing(img):
        img_np = np.array(img)
        denoised = cv2.fastNlMeansDenoising(img_np, None, h=10, 
            templateWindowSize=templateWindowSize, searchWindowSize=searchWindowSize)
        output = Image.fromarray(denoised)
        return output
    return processing

def denoise_medianBlur(kernel_size=3):
    def processing(img):
        img_np = np.array(img)
        median_filtered = cv2.medianBlur(img_np, kernel_size)
        output = Image.fromarray(median_filtered)
        return output
    return processing

def denoise_bilateralFilter():
    def processing(img):
        img_np = np.array(img)
        smoothed = cv2.bilateralFilter(img_np, d=9, sigmaColor=75, sigmaSpace=75)
        output = Image.fromarray(smoothed)
        return output
    return processing

def edge_unsharpMask():
    def processing(img):
        return img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    return processing

def edge_laplacian(alpha = 0.5):
    def processing(img):
        img_np = np.array(img)
        laplacian = cv2.Laplacian(img_np, cv2.CV_64F)
        laplacian = cv2.convertScaleAbs(laplacian)
        enhanced = cv2.addWeighted(img_np, 1.0, laplacian, alpha, 0)
        output = Image.fromarray(enhanced)
        return output
    return processing

def contrast_clahe(smooth=False, clipLimit=2.0):
    def processing(img):
        img_np = np.array(img)
        
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(8,8))
        clahe_img = clahe.apply(img_np)

        if smooth:
            clahe_img = cv2.GaussianBlur(clahe_img, (3, 3), 0)
        
        output = Image.fromarray(clahe_img)

        return output
    return processing

def contrast_gamma_correction(gamma = 0.8):
    def processing(img):
        lut = [int((i / 255.0) ** gamma * 255) for i in range(256)]
        img_out = img.point(lut)
        return img_out
    return processing