import PIL
import numpy as np
import sys
import random
import torch
from torchvision import transforms
from torchvision.transforms import functional as F
import numbers
import cv2
from source_target_transforms import *

class DataSampler:
    def __init__(self, img, sr_factor, crop_size):
        """
        Args:
            img:
            sr_factor: Interpolation factor.
            crop_size: random crop
        """
        self.img = img
        self.sr_factor = sr_factor
        self.pairs = self.create_hr_lr_pairs()
        sizes = np.float32([x[0].size[0]*x[0].size[1] / float(img.size[0]*img.size[1]) \
            for x in self.pairs])  # LRi在原图基础上被缩放的比例
        self.pair_probabilities = sizes / np.sum(sizes)

        # 旋转四个角度：0，90，180，270；随机水平翻转；随机垂直翻转；crop
        self.transform = transforms.Compose([
            RandomRotationFromSequence([0, 90, 180, 270]),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            RandomCrop(crop_size),
            ToTensor()]) 

    def create_hr_lr_pairs(self):
        # 创建HR-LR对
        # return：[tuple, tuple, ...]
        smaller_side = min(self.img.size[0 : 2])  # 最小边
        larger_side = max(self.img.size[0 : 2])  # 最大边

        factors = []  # factors包含了所有满足条件的缩放比例，这些比例可以用于下采样原始图像以生成对应的高分辨率和低分辨率图像对。
        for i in range(smaller_side//5, smaller_side+1):
            downsampled_smaller_side = i
            zoom = float(downsampled_smaller_side)/smaller_side
            downsampled_larger_side = round(larger_side*zoom)
            if downsampled_smaller_side%self.sr_factor==0 and \
                downsampled_larger_side%self.sr_factor==0:
                factors.append(zoom)

        pairs = []
        for zoom in factors:
            # 使用双三次下采样方法，获得HR
            hr = self.img.resize((int(self.img.size[0]*zoom), \
                                int(self.img.size[1]*zoom)), \
                resample=PIL.Image.BICUBIC)

            # 由于model的输入输出尺寸相同，因此LR还要resize到与HR尺寸相同
            lr = hr.resize((int(hr.size[0]/self.sr_factor), \
                int(hr.size[1]/self.sr_factor)),
                resample=PIL.Image.BICUBIC)

            lr = lr.resize(hr.size, resample=PIL.Image.BICUBIC)

            pairs.append((hr, lr))

        return pairs

    def generate_data(self):
        while True:
            # 根据pair_probabilities，随机选择k=1个数的pair作为返回值
            hr, lr = random.choices(self.pairs, weights=self.pair_probabilities, k=1)[0]
            hr_tensor, lr_tensor = self.transform((hr, lr))  # 数据增强
            hr_tensor = torch.unsqueeze(hr_tensor, 0)
            lr_tensor = torch.unsqueeze(lr_tensor, 0)
            # yield用于定义生成器函数。这样的函数在被调用时，不会立即执行，而是返回一个迭代器。当你遍历这个迭代器（例如，通过next()函数或在for循环中）时，函数将执行到下一个yield语句，然后返回yield后面的值，并保持函数的状态。在下一次迭代时，函数将从上次yield的位置继续执行。
            yield hr_tensor, lr_tensor

if __name__ == '__main__':
    img = PIL.Image.open(sys.argv[1])
    sampler = DataSampler(img, 2)
    for x in sampler.generate_data():
        hr, lr = x
        hr = hr.numpy().transpose((1, 2, 0))
        lr = lr.numpy().transpose((1, 2, 0))