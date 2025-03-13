import torch
from torchvision import models
import torch.nn as nn
from model_module import PSConv, FSAS_DFFN, MSCA, HRAMi_DRAMiT, MASAG, DTAB_GCSA, LCA, SSA, EAGFM, BIE_BIEF, SHViTBlock

# 风车卷积
PSConv1 = PSConv.PSConv(1024, 2048)
PSConv2 = PSConv.PSConv(2048, 1024)

# 图像增强
FSASDFFN = FSAS_DFFN.FSASDFFN(64)
DRAMiT = HRAMi_DRAMiT.DRAMiT(64)
MASAG_block = MASAG.MASAG(64)  # 医学特征融合
IEL = LCA.IEL(64)

# 注意力机制
DTAB_block = DTAB_GCSA.DTAB(64)
SSA_block = SSA.SSA(1024)


# 特征融合
EAGFM_block = EAGFM.EAGFM(1024)
BIEF_block = BIE_BIEF.BIEF(1024)
SHViT_Block = SHViTBlock.SHViTBlock(1024,type='s')

MSCA = MSCA.MSCAttention(1024)
model_base = models.resnet152(weights=models.ResNet152_Weights)
model_base_2 = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

class AlternativeNet(nn.Module):
    def __init__(self):
        super(AlternativeNet, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(1280, 1024, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.upsample(x)
        return self.conv(x)

class TOpNet(nn.Module):
    def __init__(self):
        super(TOpNet, self).__init__()
        self.prex = nn.Sequential(
            model_base.conv1,
            model_base.bn1,
            model_base.relu,
            model_base.maxpool,

        )  # -->[1, 64, 56, 56]
        self.eff = model_base_2.features
        self.AlternativeNet = AlternativeNet()
        self.FSASDFFN = FSASDFFN
        self.DRAMiT = DRAMiT
        self.IEL = IEL      # 图像增强
        self.MASAG = MASAG_block
        self.DTAB = DTAB_block      # 注意力模块
        self.SHViT_Block = SHViT_Block
        self.conv = nn.Sequential(
            model_base.layer1,
            model_base.layer2,      # -->[1, 512, 28, 28]
            model_base.layer3)     # -->[1, 1024, 14, 14]
        self.EAGFM = EAGFM_block
        self.strength_block = nn.Sequential(
            SSA_block,  # 注意力模块
            PSConv1,
            PSConv2,  # 风车状卷积
            MSCA,  # 医学图像多尺度交叉轴注意力模块
            model_base.layer4,
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, 8),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        x4 = self.eff(x)
        x = self.prex(x)
        x = self.IEL(x)
        x1 = self.FSASDFFN(x)
        x2 = self.DRAMiT(x)
        x = self.MASAG(x1, x2)
        x = self.DTAB(x)
        x3 = self.conv(x)
        x4 = self.AlternativeNet(x4)
        x = self.EAGFM(x3, x4)
        x = self.SHViT_Block(x)
        x = self.strength_block(x)
        return x


if __name__ == "__main__":
    device = "cuda:0"
    model = TOpNet().to(device)
    inputs = torch.rand(1, 3, 224, 224).to(device)
    out = model(inputs)
    print(out)
    print(out.size())
    print(model)
    total_params = count_parameters(model)
    print(f"Total Parameters: {total_params:,}")
