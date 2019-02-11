import torch
import numpy as np
import cv2

from config import *


def custom_loss(tensorlist):
    loss = 0.0
    for index, v in enumerate(tensorlist):
        scaling = 1.0 * v.numel()

        loss += config_coeff[index] * torch.sum(v[:, :, 2:-2, 2:-2]**2) / scaling
    return loss


def preprocessing(img):
    # img type: numpy array channel first [C x H x W]

    result = np.zeros(img.shape)

    result[0, :, :] = (img[0, :, :] - 0.485) /  0.229
    result[1, :, :] = (img[1, :, :] - 0.456) /  0.224
    result[2, :, :] = (img[2, :, :] - 0.406) /  0.225

    return result


def deprocessing(tensor):
    # tensor type: torch float tensor [0.0 ~ 1.0]

    image = tensor.cpu().detach().numpy()             # channel last
    image[0] = np.clip(image[0] * 0.229 + 0.485, 0.0, 1.0)
    image[1] = np.clip(image[1] * 0.224 + 0.456, 0.0, 1.0)
    image[2] = np.clip(image[2] * 0.225 + 0.406, 0.0, 1.0)

    image = image.transpose(1, 2, 0)
    return np.flip(image, 2)


def totensor(img_array):
    image = 1.0 * img_array #/ 255.0  # channel first
    print(image.shape)
    image = torch.from_numpy(image).unsqueeze(0)
    return image.to(device)


def upscale(img, size):
    image = img.squeeze().cpu().detach().numpy().transpose(1, 2, 0)
    image = cv2.resize(image, size)
    image = image.transpose(2, 0, 1)
    image = torch.from_numpy(image).unsqueeze(0)
    return image.to(device)
