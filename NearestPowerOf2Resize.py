import argparse
import os
import PIL
from PIL import Image

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--power', type=int, default=1, help='Power of 2')
    parser.add_argument('--imgs', type=str, help='Path to input imgs file')
    parser.add_argument('--output', type=str, help='Path to output imgs file')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = get_args()

    hr_files = sorted(os.listdir(args.imgs))

    factor = 2**args.power

    for file in hr_files:
        print("Processing: "+file)

        path_to_image = os.path.join(args.imgs, file)
        image = Image.open(path_to_image).convert('RGB')

        new_width = image.width - image.width % factor
        new_height = image.height - image.height % factor

        image = image.resize((new_width, \
            new_height), resample=PIL.Image.BICUBIC)
        path_to_save = os.path.join(args.output, file)
        image.save(path_to_save)