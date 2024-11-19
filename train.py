import numpy as np
import pyiqa
from net import ZSSRNet
from data import DataSampler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import init
import PIL
import sys
from torchvision import transforms
import tqdm
import argparse

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def adjust_learning_rate(optimizer, new_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

input_image = None

def train(model, img, sr_factor, num_batches, learning_rate, crop_size):
    loss = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    sampler = DataSampler(img, sr_factor, crop_size)  # 得到了99个HR-LR pairs

    model.to(device)

    with tqdm.tqdm(total=num_batches, miniters=1, mininterval=0) as progress:
        for iter, (hr, lr) in enumerate(sampler.generate_data()):
            model.zero_grad()

            # lr = Variable(lr).cuda()
            # hr = Variable(hr).cuda()
            lr = Variable(lr).to(device)
            hr = Variable(hr).to(device)

            output = model(lr) + lr  # 预测的是residual，而不是SR图像
            error = loss(output, hr)

            # cpu_loss = error.data.cpu().numpy()[0]
            cpu_loss = error.data.cpu().numpy()

            progress.set_description("Iteration: {iter} Loss: {loss}, Learning Rate: {lr}".format( \
                iter=iter, loss=cpu_loss, lr=learning_rate))
            progress.update()

            if iter > 0 and iter % 10000 == 0:
                learning_rate = learning_rate / 10
                adjust_learning_rate(optimizer, new_lr=learning_rate)
                print("Learning rate reduced to {lr}".format(lr=learning_rate) )

            error.backward()
            optimizer.step()

            if iter > num_batches:
                print('Done training.')
                break

def test(model, img, sr_factor):
    model.eval()

    # 由于model的输入输出尺寸相等，因此要将test image扩大为原来的两倍，才能得到尺寸更大的SR图像
    # 使用双三次插值法来扩大test image
    img = img.resize((int(img.size[0]*sr_factor), \
        int(img.size[1]*sr_factor)), resample=PIL.Image.BICUBIC)
    img.save('low_res.png')

    img = transforms.ToTensor()(img)
    img = torch.unsqueeze(img, 0)
    input = Variable(img.to(device))
    residual = model(input)
    output = input + residual

    output = output.cpu().data[0, :, :, :]
    o = output.numpy()
    # 由于是灰度图像，所以这里像素值的min=0，max=1
    o[np.where(o < 0)] = 0.0
    o[np.where(o > 1)] = 1.0

    output = torch.from_numpy(o)

    output = transforms.ToPILImage()(output) 
    
    output.save('zssr.png')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_batches', type=int, default=15000, \
        help='Number of batches to run')
    parser.add_argument('--crop', type=int, default=128, \
        help='Random crop size')
    parser.add_argument('--lr', type=float, default=0.00001, \
        help='Base learning rate for Adam')
    parser.add_argument('--factor', type=int, default=2, \
        help='Interpolation factor.')
    parser.add_argument('--img', type=str, help='Path to input img')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = get_args()

    img = PIL.Image.open(args.img).convert("RGB")
    #img = PIL.Image.open("examples/lincoln.png")
    num_channels = len(np.array(img).shape)
    if num_channels == 3:
        model = ZSSRNet(input_channels = 3)
    elif num_channels == 2:
        model = ZSSRNet(input_channels = 1)
    else:
        print("Expecting RGB or gray image, instead got", img.size)
        sys.exit(1)

    # Weight initialization
    model.apply(weights_init_kaiming)  # 为了解决梯度消失/爆炸问题，将所有conv和linear进行KaiMing初始化。其原理是保持每一层的输入和输出的方差一致。

    input_image = np.array(img)

    train(model, img, args.factor, args.num_batches, args.lr, args.crop)
    test(model, img, args.factor)
    #train(model, img, 2, 15000, 0.00001, 128)
    #test(model, img, 2)

    iqa_ILNIQE = pyiqa.create_metric('ilniqe', device=device)
    iqa_NIQE = pyiqa.create_metric('niqe', device=device)
    prior_ILNIQE = iqa_ILNIQE('./low_res.png')
    prior_NIQE = iqa_NIQE('./low_res.png')
    after_ILNIQE = iqa_ILNIQE('./zssr.png')
    after_NIQE = iqa_NIQE('./zssr.png')
    print(f"Score Prior: " +
            f"\n\t{prior_NIQE.item()} (NIQE), " +
            f"\n\t{prior_ILNIQE.item()} (ILNIQE)")
    print(f"Score After: " +
            f"\n\t{after_NIQE.item()} (NIQE), " +
            f"\n\t{after_ILNIQE.item()} (ILNIQE)")

