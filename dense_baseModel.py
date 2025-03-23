import torch
from torchvision import models
import torch.nn as nn
from model_module import PCAA, PSConv


PCAA_block = PCAA.PCAA(3)
vit = models.vit_b_16(weights=models.ViT_B_16_Weights)
vit.heads.head = nn.Linear(vit.heads.head.in_features, 8)
class TOpNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super(TOpNet, self).__init__(*args, **kwargs)
        self.PCAA_block = PCAA_block
        self.backbone = models.densenet169(weights=models.DenseNet169_Weights.IMAGENET1K_V1).features
        self.vit = vit
        self.fc_layers = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1664, 1024),
            nn.Dropout(0.6),
            nn.Linear(1024, 256),
            nn.Dropout(0.6),
            nn.Linear(256, 8),
            nn.Dropout(0.6),
        )
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = PCAA_block(x)
        x1 = vit(x)
        x= self.backbone(x)
        x = self.fc_layers(x)
        x = self.sigmoid(x) * x1 + self.sigmoid(x1) * x
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    device = "cuda:0"
    model = TOpNet()
    inputs = torch.rand(1, 3, 224, 224)
    out = model(inputs)
    # print(out)
    print(out.size())
    # print(model)
    total_params = count_parameters(model)
    print(f"Total Parameters: {total_params:,}")
