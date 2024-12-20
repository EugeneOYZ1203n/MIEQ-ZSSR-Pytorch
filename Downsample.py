import argparse
import os
import PIL
from PIL import Image

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--factor', type=int, default=4, help='Scale factor')
    parser.add_argument('--imgs', type=str, help='Path to input hr imgs file')
    parser.add_argument('--output', type=str, help='Path to output imgs file')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = get_args()

    hr_files = sorted(os.listdir(args.imgs))

    for file in hr_files:
        print("Processing: "+file)

        path_to_image = os.path.join(args.imgs, file)
        image = Image.open(path_to_image).convert('RGB')

        image = image.resize((int(image.size[0]/args.factor), \
            int(image.size[1]/args.factor)), resample=PIL.Image.BICUBIC)
        path_to_save = os.path.join(args.output, file)
        image.save(path_to_save)