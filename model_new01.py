import torch
from torchvision import models
import torch.nn as nn
from model_module import PSConv, FSAS_DFFN, MSCA, HRAMi_DRAMiT, MASAG, DTAB_GCSA, LCA, SSA

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

MSCA = MSCA.MSCAttention(1024)
model_base = models.resnet152()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


class TOpNet(nn.Module):
    def __init__(self):
        super(TOpNet, self).__init__()
        self.prex = nn.Sequential(
            model_base.conv1,
            model_base.bn1,
            model_base.relu,
            model_base.maxpool
        )  # -->[1, 64, 56, 56]
        self.FSASDFFN = FSASDFFN
        self.DRAMiT = DRAMiT
        self.IEL = IEL
        self.MASAG = MASAG_block
        self.DTAB = DTAB_block      # 注意力模块
        self.conv = nn.Sequential(
            model_base.layer1,
            model_base.layer2,
            model_base.layer3,      # -->[1, 1024, 14, 14]
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
        x = self.prex(x)
        x = self.IEL(x)
        x1 = self.FSASDFFN(x)
        x2 = self.DRAMiT(x)
        x = self.MASAG(x1, x2)
        x = self.DTAB(x)
        x = self.conv(x)
        return x


if __name__ == "__main__":
    device = "cuda:0"
    model = TOpNet().to(device)
    inputs = torch.rand(1, 3, 224, 224).to(device)
    out = model(inputs)
    print(out)
    print(out.size())
    # print(model)
    total_params = count_parameters(model)
    print(f"Total Parameters: {total_params:,}")
