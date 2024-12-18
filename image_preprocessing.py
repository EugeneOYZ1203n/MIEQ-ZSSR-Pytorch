from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np

def basic_denoise_sharpen_contrast(img):
    img_denoised = img.filter(ImageFilter.GaussianBlur(radius=1))
    img_sharpened = img_denoised.filter(ImageFilter.SHARPEN)

    enhancer = ImageEnhance.Contrast(img_sharpened)
    img_contrast = enhancer.enhance(1.2)

    return img_contrast

def denoise_fastN1MeansDenoising(img):
    img_np = np.array(img)
    denoised = cv2.fastNlMeansDenoising(img_np, None, h=10, templateWindowSize=7, searchWindowSize=21)
    output = Image.fromarray(denoised)
    return output

def denoise_medianBlur(img):
    img_np = np.array(img)
    median_filtered = cv2.medianBlur(img_np, 3)
    output = Image.fromarray(median_filtered)
    return output

def denoise_bilateralFilter(img):
    img_np = np.array(img)
    smoothed = cv2.bilateralFilter(img_np, d=9, sigmaColor=75, sigmaSpace=75)
    output = Image.fromarray(smoothed)
    return output

def edge_unsharpMask(img):
    return img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

def edge_laplacian(img, alpha = 1.0):
    img_np = np.array(img)
    laplacian = cv2.Laplacian(img_np, cv2.CV_64F)
    laplacian = cv2.convertScaleAbs(laplacian)
    enhanced = cv2.addWeighted(img_np, 1.0, laplacian, alpha, 0)
    output = Image.fromarray(enhanced)
    return output

def contrast_clahe(img, smooth=False):
    img_np = np.array(img)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(img_np)

    if smooth:
        clahe_img = cv2.GaussianBlur(clahe_img, (3, 3), 0)
    
    output = Image.fromarray(clahe_img)

    return output

def contrast_gamma_correction(img, gamma = 0.8):
    lut = [int((i / 255.0) ** gamma * 255) for i in range(256)]
    img_out = img.point(lut)
    return img_out