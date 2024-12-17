import argparse
import os

import PIL
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
from train import train_model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batches', type=int, default=15000, \
        help='Number of batches to run')
    parser.add_argument('--crop', type=int, default=128, \
        help='Random crop size')
    parser.add_argument('--lr', type=float, default=0.00001, \
        help='Base learning rate for Adam')
    parser.add_argument('--factor', type=int, default=2, \
        help='Interpolation factor.')
    parser.add_argument('--imgs', type=str, help='Path to input imgs file')
    parser.add_argument('--output', type=str, help='Path to output imgs file')

    args = parser.parse_args()

    return args

def downsample(img, factor, out_path):
    img = img.resize((int(img.size[0]/factor), \
        int(img.size[1]/factor)), resample=PIL.Image.BICUBIC)
    img.save(out_path)

    return img

def image_preprocessing(img, out_path):
    output = clahe(img)
    output.save(out_path)

    return output

def denoise_sharpen_contrast(img):
    img_denoised = img.filter(ImageFilter.GaussianBlur(radius=1))
    img_sharpened = img_denoised.filter(ImageFilter.SHARPEN)

    enhancer = ImageEnhance.Contrast(img_sharpened)
    img_contrast = enhancer.enhance(1.2)

    return img_contrast

def edge_enhancement(img):
    return img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

def clahe(img):
    img_np = np.array(img)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(img_np)
    
    # Convert back to Pillow image
    output = Image.fromarray(clahe_img)

    return output

if __name__ == '__main__':
    args = get_args()

    gt_files = sorted(os.listdir(args.imgs))

    print("Found images:" + ",".join(gt_files))

    for i, img in enumerate(gt_files):
        ## Downsample all images
        img_path = os.path.join(args.imgs, img)
        gt_img = PIL.Image.open(img_path).convert('L')

        out_path = os.path.join(args.output, "lr_" + img)
        lr_img = downsample(gt_img, args.factor, out_path)

        ## Apply preprocessing
        out_path = os.path.join(args.output, "pp_" + img)
        pp_img = image_preprocessing(lr_img, out_path)

        ## Generate zssr
        out_path = os.path.join(args.output, "sr_" + img)
        train_model(pp_img, args.factor, args.batches, args.lr, args.crop, out_path)
        

