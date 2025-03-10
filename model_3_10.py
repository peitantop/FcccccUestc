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
        FSAS_DFFN,
        model_base.layer1,
        model_base.layer2,
        model_base.layer3,
        PSConv1,
        PSConv2,
        MSCA,
        model_base.layer4,
        nn.AdaptiveMaxPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(2048, 8),
        nn.Sigmoid()
    )
        

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == "__main__":
    model = TOpNet()
    inputs = torch.rand(1, 3, 224, 224)
    out = model(inputs)
    print(out)
    print(out.size())
    # print(model)
    total_params = count_parameters(model)
    print(f"Total Parameters: {total_params:,}")