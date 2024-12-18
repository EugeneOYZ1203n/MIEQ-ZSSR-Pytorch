import argparse
import os
import PIL
from image_preprocessing import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, help='Path to input img')
    parser.add_argument('--output', type=str, help='Path to output img folder')

    args = parser.parse_args()

    return args

preprocessingFuncs = {
    "Basic": [basic_denoise_sharpen_contrast],
    "Unsharp Mask": [edge_unsharpMask],
    "Laplacian": [edge_laplacian],
    "CLAHE": [contrast_clahe],
    "Gamma Correction": [contrast_gamma_correction],
    "FastN1Means+Unsharp Mask": [denoise_fastN1MeansDenoising, edge_unsharpMask],
    "Median Blur+Unsharp Mask": [denoise_medianBlur, edge_unsharpMask],
    "Bilateral Filter+Unsharp Mask": [denoise_bilateralFilter, edge_unsharpMask],
    "FastN1Means+Laplacian": [denoise_fastN1MeansDenoising, edge_laplacian],
    "Median Blur+Laplacian": [denoise_medianBlur, edge_laplacian],
    "Bilateral Filter+Laplacian": [denoise_bilateralFilter, edge_laplacian],
    "FastN1Means+CLAHE": [denoise_fastN1MeansDenoising, contrast_clahe],
    "Median Blur+CLAHE": [denoise_medianBlur, contrast_clahe],
    "Bilateral Filter+CLAHE": [denoise_bilateralFilter, contrast_clahe],
    "FastN1Means+Gamma Correction": [denoise_fastN1MeansDenoising, contrast_gamma_correction],
    "Median Blur+Gamma Correction": [denoise_medianBlur, contrast_gamma_correction],
    "Bilateral Filter+Gamma Correction": [denoise_bilateralFilter, contrast_gamma_correction],
}

if __name__ == '__main__':
    args = get_args()

    gt_img = PIL.Image.open(args.img).convert('L')
    extension = args.img.split(".")[-1]

    for i in preprocessingFuncs.keys():
        output = gt_img
        for f in preprocessingFuncs[i]:
            output = f(output)
        out_path = os.path.join(args.output, i.replace(" ", "_") + "." + extension)
        output.save(out_path)

