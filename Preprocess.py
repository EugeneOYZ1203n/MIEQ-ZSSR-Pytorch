import argparse
import os
import PIL
from PIL import Image

from image_preprocessing import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgs', type=str, help='Path to input hr imgs file')
    parser.add_argument('--output', type=str, help='Path to output imgs file')

    args = parser.parse_args()

    return args

def image_preprocessing(img):
    #img = denoise_fastN1MeansDenoising(5, 7)(img)
    #img = contrast_clahe(clipLimit=3.0)(img)
    #img = edge_unsharpMask()(img)
    img = misc_normalise_img()(img)

    return img

if __name__ == '__main__':
    args = get_args()

    hr_files = sorted(os.listdir(args.imgs))

    for file in hr_files:
        print("Processing: "+file)

        path_to_image = os.path.join(args.imgs, file)
        image = Image.open(path_to_image).convert('L')

        image = image_preprocessing(image)

        path_to_save = os.path.join(args.output, file)
        image.save(path_to_save)