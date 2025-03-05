from typing import Optional, Callable, Type, Union, List

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import models
import torch.nn as nn
from torchvision.utils import _log_api_usage_once
import segmentation_models_pytorch as smp
from model_module import EAGFM, HFF_MSFA



unet = smp.Unet(
    encoder_name="resnet50",
    encoder_depth=5,
    encoder_weights=None,
    decoder_attention_type="scse",
    activation="sigmoid"
)
densenet = models.densenet201()
resnet = models.resnet152()


class TanNet(nn.Module):
    def __init__(self):
        super(TanNet, self).__init__()
        self.resnet_features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )

        self.unet = unet

        self.densenet_features = nn.Sequential(
            densenet.features,
            nn.Conv2d(1920, 2048, kernel_size=1, stride=1)
        )

        self.layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),  # [1, 64, 112, 112]
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # [1, 64, 56, 56]
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # [1, 128, 28, 28]
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # [1, 256, 14, 14]
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # [1, 512, 7, 7]
            nn.Conv2d(512, 2048, kernel_size=1, stride=1)  # [1, 2048, 7, 7]
        )


    def forward(self, x):
        x1 = self.resnet_features(x)
        x2 = self.unet(x)
        x3 = self.densenet_features(x)
        x2 = self.layers(x2)
        EGAFM_module = EAGFM.EAGFM(2048)      # 初始化EAGFM模块并设定通道维度
        x4 = EGAFM_module(x1, x3)
        msfa = HFF_MSFA.MSFA(2048)
        x5 = msfa(x2, x4)
        return x5



if __name__ == "__main__":
    model = TanNet()
    inputtensor = torch.rand(1, 3, 224, 224)
    output = model(inputtensor)
    print(output.size())
