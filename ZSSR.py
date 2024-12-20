import argparse
import os
import PIL
from train import train_model
from image_preprocessing import *


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

if __name__ == '__main__':
    args = get_args()

    files = sorted(os.listdir(args.imgs))

    print("Found images:" + ",".join(files))

    for i, img in enumerate(files):
        print("\n\nProcessing: " + img)

        ## Downsample all images
        img_path = os.path.join(args.imgs, img)
        img = PIL.Image.open(img_path).convert('L')

        ## Generate zssr
        out_path = os.path.join(args.output, img)
        train_model(img, args.factor, args.batches, args.lr, args.crop, out_path)
        

