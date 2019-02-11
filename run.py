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


def gradascent(img, model, step=0.005, max_iteration=iteration, max_loss=max_loss):
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
            img = img + torch.clamp(grads_norm, -2.0, 2.0) * step

    return img


if __name__ == '__main__':
    # initialization ------------------
    image_path = '1.png'
    ori_image = cv2.imread(image_path, -1)
    height, width = ori_image.shape[0], ori_image.shape[1]

    # image = 1.0 * cv2.imread(image_path, -1).transpose(2, 0, 1)/255.0  # channel first


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
        if scale == 0:
            real_img = 1.0*cv2.resize(ori_image, image_shape_list[scale][::-1]).transpose(2, 0, 1)/255.0
            image = preprocessing(real_img)
            input_img = totensor(image)

        input_img = Variable(input_img, requires_grad=True).to(device).float()
        dream_img = gradascent(input_img, net)

        print(dream_img.size())
        if scale < 2:
            ups_dream_img = upscale(dream_img, size=image_shape_list[scale+1][::-1])
            real_img_up = 1.0*cv2.resize(ori_image, image_shape_list[scale+1][::-1]).transpose(2, 0, 1)/255.0
            real_img_up = totensor(preprocessing(real_img_up)).float()
            real_img_cur= cv2.resize(ori_image, image_shape_list[scale][::-1])
            real_img_cur = 1.0 * cv2.resize(ori_image, image_shape_list[scale+1][::-1]).transpose(2, 0, 1)/255.0
            real_img_cur = totensor(preprocessing(real_img_cur)).float()

            lost_detail = real_img_up - real_img_cur
            ups_dream_img += lost_detail
            input_img = ups_dream_img.clone()

        saved_img = deprocessing(input_img.squeeze())
        cv2.imwrite('saved_{}.png'.format(scale), (255*saved_img).astype('uint8'))
    # dream_img = gradascent(input_img, net)
    #


