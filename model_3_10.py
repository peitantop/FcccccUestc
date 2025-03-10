import torch
from torchvision import models
import torch.nn as nn
from model_module import PSConv, FSAS_DFFN, MSCA

PSConv1 = PSConv.PSConv(1024, 2048)
PSConv2 = PSConv.PSConv(2048, 1024)
FSAS_DFFN = FSAS_DFFN.FSASDFFN(64)
MSCA = MSCA.MSCAttention(1024)
model_base = models.resnet152()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


class TOpNet(nn.Module):
    def __init__(self):
        super(TOpNet,self).__init__()
        self. model = nn.Sequential(
        model_base.conv1,
        model_base.bn1,
        model_base.relu,
        model_base.maxpool,
        FSAS_DFFN,  # 图像去模糊去噪  --> [1, 64, 56, 56]
        model_base.layer1,
        model_base.layer2,
        model_base.layer3,
        PSConv1,
        PSConv2,    # 风车状卷积
        MSCA,       # 医学图像多尺度交叉轴注意力模块
        model_base.layer4,
        nn.AdaptiveMaxPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(2048, 8),
        nn.Dropout(0.5)
    )
        
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
        

    def forward(self, x):
        x = self.model(x)
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

    # # 添加模型输出范围检查
    # with torch.no_grad():
    #     sample = torch.randn(2,3,224,224).to(device)
    #     outputs = model(sample)
    #     print("Output range:", torch.sigmoid(outputs).min().item(), torch.sigmoid(outputs).max().item())