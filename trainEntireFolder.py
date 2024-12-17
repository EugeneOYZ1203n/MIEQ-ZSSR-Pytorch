import argparse
import os

import PIL
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

def downsample(img_name, factor, out_path):
    img = PIL.Image.open(img_name).convert('L')

    img = img.resize((int(img.size[0]/factor), \
        int(img.size[1]/factor)), resample=PIL.Image.BICUBIC)
    img.save(out_path)

if __name__ == '__main__':
    args = get_args()

    gt_files = sorted(os.listdir(args.imgs))

    print("Found images:" + ",".join(gt_files))

    lr_files = []

    ## Downsample all images
    for i, img in enumerate(gt_files):
        out_name = "lr_" + img
        img_path = os.path.join(args.imgs, img)
        out_path = os.path.join(args.imgs, out_name)
        downsample(img_path, args.factor, out_path)
        lr_files.append(out_name)
    
    ## Generate ZSSR images
    for i, img in enumerate(lr_files):
        img_path = os.path.join(args.imgs, img)
        lr_img = PIL.Image.open(img_path).convert("L")
        out_name = "sr_" + img[3:]
        out_path = os.path.join(args.imgs, out_name)
        train_model(lr_img, args.factor, args.batches, args.lr, args.crop, out_path)

