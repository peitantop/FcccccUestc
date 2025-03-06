from typing import Optional, Callable, Type, Union, List

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import models
import torch.nn as nn
from torchvision.utils import _log_api_usage_once
import segmentation_models_pytorch as smp
from model_module import EAGFM, HFF_MSFA,FCA, HRAMi_DRAMiT, DTAB_GCSA, FSAS_DFFN, IGAB



unet = smp.Unet(
    encoder_name="resnet50",
    encoder_depth=5,
    encoder_weights=None,
    decoder_attention_type="scse",
    activation="sigmoid"
)
densenet = models.densenet201()
resnet = models.resnet152()
ImageStrength_block1 = FCA.FCAttention(3)
ImageStrength_block2 = HRAMi_DRAMiT.DRAMiT(dim=64, num_head=64)
ImageStrength_block3 =  DTAB_GCSA.DTAB(64)
ImageStrength_block4 = DTAB_GCSA.GCSA(64)
ImageStrength_block5 = FSAS_DFFN.FSASDFFN(64)
ImageStrength_block6 = IGAB.IGAB(64,dim_head=64)
class TanNet(nn.Module):
    def __init__(self):
        super(TanNet, self).__init__()
        self.ImageStrength_block1 = ImageStrength_block1
        self.ImageStrength_block2 = ImageStrength_block2
        self.ImageStrength_block3 = ImageStrength_block3
        self.ImageStrength_block4 = ImageStrength_block4
        self.ImageStrength_block5 = ImageStrength_block5
        self.ImageStrength_block6 = ImageStrength_block6

        self.sigmoid = nn.Sigmoid()
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
        self.changeLayer1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.changeLayer2 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),  # [1, 64, 112, 112]
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # [1, 64, 56, 56]
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # [1, 128, 28, 28]
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # [1, 256, 14, 14]
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # [1, 512, 7, 7]
            nn.Conv2d(512, 2048, kernel_size=1, stride=1)  # [1, 2048, 7, 7]
        )
        self.EGAFM = EAGFM.EAGFM(2048)  
        self.msfa = HFF_MSFA.MSFA(2048)


    def forward(self, x):
        x1 = self.ImageStrength_block1(x)
        x1 = self.changeLayer1(x1)
        x2 = self.ImageStrength_block2(x1)
        x = self.changeLayer1(x)
        x3 = self.ImageStrength_block3(x)
        x4 = self.ImageStrength_block4(x)
        x5 = self.sigmoid(x4) * x3
        x6 = self.ImageStrength_block5(x5)
        x7 = self.ImageStrength_block6(x2, x6)
        x7 = self.changeLayer2(x7)
        x8 = self.resnet_features(x7)
        x9 = self.unet(x7)
        x10 = self.densenet_features(x7)
        x11 = self.layers(x9)    
        x12 = self.EGAFM(x8, x10)
        x13 = self.msfa(x11, x12)
        return x13



if __name__ == "__main__":
    model = TanNet()
    inputtensor = torch.rand(1, 3, 224, 224)
    output = model(inputtensor)
    print(output)
    print(output.size())
