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
    encoder_name="resnet101",
    encoder_depth=5,
    encoder_weights="imagenet",
    decoder_attention_type="scse",
    activation="sigmoid"
)

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
            2048,1024, kernel_size=3, 
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
class AdaptivePoolingClassifier(nn.Module):
    def __init__(self, input_dim, num_labels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(input_dim*2, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_labels))
        
    def forward(self, x):
        avg_x = self.avg_pool(x).squeeze(-1).squeeze(-1)
        max_x = self.max_pool(x).squeeze(-1).squeeze(-1)
        combined = torch.cat([avg_x, max_x], dim=1)
        return self.fc(combined)

class TannetLigtht(nn.Module):
    def __init__(self, num_classes):
        super(TannetLigtht, self).__init__()
        self.bn1 = nn.BatchNorm2d(64)
        # self.bn2 = nn.BatchNorm2d(2048)
        self.unet = unet
        self.DRAMIT = HRAMi_DRAMiT.DRAMiT(dim=64, num_head=64)
        self.changeLayer1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.changeLayer2 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.RHmodel = RHmodel
        self.unetlayers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),  # [1, 64, 112, 112]
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # [1, 64, 56, 56]
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # [1, 128, 28, 28]
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # [1, 256, 14, 14]
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # [1, 512, 7, 7]
            nn.Conv2d(512, 2048, kernel_size=1, stride=1)  # [1, 2048, 7, 7]
        )
        self.PSConv = PSConv.PSConv(1024, 2048)
        self.MSCA = MSCA.MSCAttention(2048)
        self.classifier = AdaptivePoolingClassifier(
            input_dim=2048,  # 根据实际特征维度调整
            num_labels=num_classes
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.changeLayer1(x)
        x = F.relu(self.bn1(x))
        x = self.DRAMIT(x)
        x = self.changeLayer2(x)
        # x = F.relu(self.bn2(x))
        x = self.unet(x)
        x = self.unetlayers(x)
        x = self.RHmodel.encoder(x)
        x = self.PSConv(x)
        x = self.MSCA(x)
        x = self.RHmodel.decoder(x)
        logits = self.classifier(x)
        return logits
   


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

if __name__ == "__main__":
    model = TannetLigtht(8)
    inputtensor = torch.rand(1, 3, 224, 224)
    output = model(inputtensor)
    print(output)
    print(output.size())
    total_params = count_parameters(model)
    print(f"Total Parameters: {total_params:,}")