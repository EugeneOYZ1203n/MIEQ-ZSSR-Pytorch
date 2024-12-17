import argparse
import os
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
from skimage.metrics import structural_similarity as calculate_ssim
from PIL import Image
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', type=str, help='Path to ground truth files')
    parser.add_argument('--sr', type=str, help='Path to super resolution files')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = get_args()

    ground_truth_dir = args.gt
    sr_output_dir = args.sr

    psnr_values = []
    ssim_values = []

    gt_files = sorted(os.listdir(ground_truth_dir))
    #gt_files = gt_files[1:2] + gt_files[7:]
    sr_files = sorted(os.listdir(sr_output_dir))

    print(gt_files, sr_files)

    for gt_file, sr_file in zip(gt_files, sr_files):
        gt_path = os.path.join(ground_truth_dir, gt_file)
        sr_path = os.path.join(sr_output_dir, sr_file)    

        sr_img = np.array(Image.open(sr_path).convert('L'))
        sr_height, sr_width = sr_img.shape[:2]

        gt_img = Image.open(gt_path).convert('L')
        gt_img_r = gt_img.resize((sr_width, sr_height), Image.BICUBIC)
        gt_img_r = np.array(gt_img_r)

        print(sr_img.shape, gt_img_r.shape)

        psnr = calculate_psnr(sr_img, gt_img_r, data_range=gt_img_r.max()-gt_img_r.min())
        ssim = calculate_ssim(sr_img, gt_img_r, 
                        win_size=11, data_range=gt_img_r.max()-gt_img_r.min())

        psnr_values.append(psnr)
        ssim_values.append(ssim)

        print(f"Processed: {gt_path} and {sr_path}, PSNR: {psnr:.4f}, SSIM: {ssim:.4f}")

    avg_psnr = sum(psnr_values) / len(psnr_values)
    avg_ssim = sum(ssim_values) / len(ssim_values)

    print(f"Avg PSNR: {avg_psnr}, Avg SSIM: {avg_ssim}")