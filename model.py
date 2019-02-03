import torch
import torch.nn as nn
from torchvision.models import inception_v3
from torchvision.models.inception import InceptionAux

from utils import custom_loss

BASEMODEL = inception_v3(pretrained=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net = nn.ModuleList()
        num_layer = 0
        for i, v in enumerate(BASEMODEL.children()):
            if isinstance(v, InceptionAux):
                break
            else:
                self.net.append(v)
                num_layer += 1
        self.total_layer = num_layer

    def forward(self, x):
        for i in range(self.total_layer-4):
            x = self.net[i](x)

        x = self.net[-4](x)
        y = self.net[-3](x)
        z = self.net[-2](x)
        k = self.net[-1](x)

        return x, y, z, k

import cv2
# unittest
if __name__ == '__main__':
    # input = torch.randn(1, 3, 400, 400)
    # net = Net()
    # result = net(input)
    # loss = custom_loss(result)
    # print(loss)
    img = cv2.imread('1.png')
    img = cv2.resize(img, (100,200))
    cv2.imwrite('11.png', img)
