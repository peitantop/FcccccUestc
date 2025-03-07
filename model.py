from typing import Optional, Callable, Type, Union, List

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import _log_api_usage_once
import segmentation_models_pytorch as smp
from model_module import EAGFM, HFF_MSFA,FCA, HRAMi_DRAMiT, DTAB_GCSA, FSAS_DFFN, IGAB, CCFF, MSCA, CVIM, CPAM, FFM, MSPA, SCSA, MASAG, PSConv



unet = smp.Unet(
    encoder_name="resnet50",
    encoder_depth=5,
    encoder_weights=None,
    decoder_attention_type="scse",
    activation="sigmoid"
)
densenet = models.densenet201()
resnet = models.resnet152()
ImageStrength_block1 = FCA.FCAttention(channel=3)
ImageStrength_block2 = HRAMi_DRAMiT.DRAMiT(dim=64, num_head=64)
ImageStrength_block3 =  DTAB_GCSA.DTAB(dim=64)
ImageStrength_block4 = DTAB_GCSA.GCSA(dim=64)
ImageStrength_block5 = FSAS_DFFN.FSASDFFN(dim=64)
ImageStrength_block6 = IGAB.IGAB(dim=64,dim_head=64)



class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 上采样部分 (7x7 -> 64x64)
        self.tconv1 = nn.ConvTranspose2d(
            2048, 1024, kernel_size=3, 
            stride=2, padding=1, output_padding=1
        )
        self.tconv2 = nn.ConvTranspose2d(
            1024, 1024, kernel_size=3,
            stride=2, padding=1, output_padding=1
        )
        self.tconv3 = nn.ConvTranspose2d(
            1024, 1024, kernel_size=3,
            stride=2, padding=1, output_padding=1
        )
        # 插值上采样调整最终尺寸
        self.upsample = nn.Upsample(
            size=(64, 64), mode='bilinear', align_corners=False
        )
        # 最后的卷积细化特征
        self.final_conv = nn.Conv2d(
            1024, 1024, kernel_size=3, padding=1
        )

    def forward(self, x):
        x = F.relu(self.tconv1(x))  # [1,1024,14,14]
        x = F.relu(self.tconv2(x))  # [1,1024,28,28]
        x = F.relu(self.tconv3(x))  # [1,1024,56,56]
        x = self.upsample(x)        # [1,1024,64,64]
        x = self.final_conv(x)      # 保持尺寸
        return x

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 下采样部分 (64x64 -> 7x7)
        self.conv1 = nn.Conv2d(
            1024, 1024, kernel_size=3, 
            stride=2, padding=1
        )
        self.conv2 = nn.Conv2d(
            1024, 2048, kernel_size=3,
            stride=2, padding=1
        )
        self.conv3 = nn.Conv2d(
            2048, 2048, kernel_size=3,
            stride=2, padding=1
        )
        # 自适应池化对齐尺寸
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7,7))

    def forward(self, x):
        x = F.relu(self.conv1(x))  # [1,1024,32,32]
        x = F.relu(self.conv2(x))  # [1,2048,16,16]
        x = F.relu(self.conv3(x))  # [1,2048,8,8]
        x = self.adaptive_pool(x)  # [1,2048,7,7]
        return x

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

RHmodel = Autoencoder()

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
        self.RHmodel = RHmodel
        self.CCFF = CCFF.CCFF(in_channels=1024, out_channels=1024)
        self.msca = MSCA.MSCAttention(dim=1024)
        self.CVIM = CVIM.CVIM(c=1024)
        self.CPAM = CPAM.CPAM(1024)
        self.FFM = FFM.FFM(1024)
        self.MSPA = MSPA.MSPAModule(inplanes=256,scale=4)
        self.SCSA = SCSA.SCSA(dim=1024, head_num=8)
        self.MASAG = MASAG.MASAG(1024)
        self.PSConv = PSConv.PSConv(1024, 1024)
        # self.AVGpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc1 = nn.Linear(2048, 2048)
        # self.fc2 = nn.Linear(2048, 2048)
        # self.fc3 = nn.Linear(2048, 2048)
        # self.fc4 = nn.Linear(2048, 8)

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
        x13 = self.RHmodel.encoder(x13)
        x13 = self.PSConv(x13)
        x14 = self.CCFF(x13)
        x17 = self.msca(x13)
        x15 = self.CVIM(x14, x17)
        x16 = self.CPAM([x14, x17])
        x18 = self.FFM(x15, x16)
        x19 = self.MSPA(x18)
        x20 = self.SCSA(x18)
        x21 = self.MASAG(x19, x20)
        x21 = self.RHmodel.decoder(x21)
        # x21 = self.AVGpool(x21)
        # x21 = self.fc1(x21)
        # x21 = self.fc2(x21)
        # x21 = self.fc3(x21)

        # x21 = self.fc4(x21)

        return x21



def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
 

if __name__ == "__main__":
    model = TanNet()
    inputtensor = torch.rand(1, 3, 224, 224)
    output = model(inputtensor)
    print(output)
    print(output.size())
    # print(model.parameters)
    total_params = count_parameters(model)
    print(f"Total Parameters: {total_params:,}")



