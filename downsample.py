import argparse
import PIL
import PIL.Image

if __name__ == '__main__':

    for i in range(1,13):
        img_name = "./secret/high_res/highres" + str(i)
        if i > 4 and i < 11:
            img_name += ".bmp"
        else:
            img_name += ".jpg"

        img = PIL.Image.open(img_name).convert("RGB")

        img = img.resize((int(img.size[0]*0.25), \
            int(img.size[1]*0.25)), resample=PIL.Image.BICUBIC)
        img.save('./secret/low_res2/lowres' + str(i) + '.jpg')