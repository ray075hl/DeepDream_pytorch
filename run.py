import torch
from torch.autograd import Variable
import numpy as np
import cv2

from model import Net
from utils import *
from config import *

import math


grads = {}
# https://discuss.pytorch.org/t/why-cant-i-see-grad-of-an-intermediate-variable/94/6
def save_grad(name):

    def hook(grad):
        grads[name] = grad
    return hook


def gradascent(img, model, step=0.01, max_iteration=iteration, max_loss=max_loss):
    for i in range(max_iteration):
        print(i)
        model.zero_grad()
        res = model(img)
        loss_ = custom_loss(res)
        print(loss_)
        img.register_hook(save_grad('input_img'))
        loss_.backward()

        grads_tmp = grads['input_img']

        grads_norm = grads_tmp / torch.max(torch.mean(torch.abs(grads_tmp)), 1e-7*torch.ones(1).to(device))
        if loss_ > max_loss:
            break
        else:
            img = img + grads_norm * step

    return img


if __name__ == '__main__':
    # initialization ------------------
    image_path = '1.png'
    ori_image = cv2.imread(image_path, -1)
    height, width = ori_image.shape[1], ori_image.shape[2]

    image = 1.0 * cv2.imread(image_path, -1).transpose(2, 0, 1)/255.0  # channel first


    image_shape_list = []

    for i in range(num_octave):
        image_shape_list.append((math.ceil(height/1.4**(num_octave-1-i)), math.ceil(width/1.4**(num_octave-1-i))))
    print(image_shape_list)
    # image = preprocessing(image)
    #
    # input_img = torch.from_numpy(image).unsqueeze(0)  # four dimension for net input
    # input_img = Variable(input_img.float(), requires_grad=True).to(device)

    net = Net().to(device)
    for scale in range(num_octave):
        real_img = cv2.resize(ori_image, image_shape_list[scale][::-1])
        image = preprocessing(real_img)
        input_img = totensor(image)
        input_img = Variable(input_img.float(), requires_grad=True).to(device)

        dream_img = gradascent(input_img, net)
        if scale < 2:
            ups_dream_img = upscale(dream_img, size=image_shape_list[scale+1][::-1])


    # dream_img = gradascent(input_img, net)
    #
    # saved_img = deprocessing(dream_img.squeeze())
    # cv2.imwrite('saved.png', (255*saved_img).astype('uint8'))

