import torch
from torchvision import models
import torch.nn as nn
from model_module import  PSConv, EUCB_ESUM
import torch.nn.functional as F
from torchsummary import summary

PSConv_block = PSConv.PSConv(1664,2048)
vit = models.vit_b_16(weights=models.ViT_B_16_Weights)
vit.heads.head = nn.Linear(vit.heads.head.in_features, 8)


class TOpNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super(TOpNet, self).__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(in_channels=256,out_channels=6,kernel_size=4,stride=2, padding=1)
        self.meum = EUCB_ESUM.MEUM(in_channels=6, out_channels=256)
        self.psconv = PSConv_block
        
        # 修改DenseNet的第一层卷积为6通道输入
        self.backbone = models.densenet169(weights=models.DenseNet169_Weights.IMAGENET1K_V1).features
        self.backbone.conv0 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 修改ViT的patch_embedding层为6通道输入
        self.vit = vit
        self.vit.conv_proj = nn.Conv2d(6, 768, kernel_size=16, stride=16)
        # 添加交叉注意力层
        self.cross_attention = nn.MultiheadAttention(embed_dim=2048, num_heads=8)
        self.norm1 = nn.LayerNorm(normalized_shape=2048)
        self.norm2 = nn.LayerNorm(normalized_shape=2048)
        
        self.fc_layers = nn.Sequential(
            nn.AdaptiveMaxPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=2048, out_features=1024),
            nn.Dropout(0.2),
            nn.Linear(in_features=1024, out_features=256),
            nn.Dropout(0.2),
            nn.Linear(in_features=256, out_features=8),
            nn.Dropout(0.2),
        )
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.meum(x)
        x = self.conv1(x)
        # ViT分支
        x1 = self.vit(x)  # [B, 8]
        x1_expanded = x1.unsqueeze(1).expand(-1, 196, -1)  # [B, 196, 8]
        
        # Backbone分支
        x = self.backbone(x)  # [B, 1664, H, W]
        x = self.psconv(x)
        B, C, H, W = x.shape
        x = x.view(B, C, -1).permute(2, 0, 1)  # [HW, B, C]
        
        # 应用交叉注意力
        x_attended, _ = self.cross_attention(
            self.norm1(x),
            self.norm2(x),
            self.norm2(x)
        )
        x = x + x_attended  # 残差连接
        x = x.permute(1, 2, 0).view(B, C, H, W)  # 恢复原始形状
        
        x = self.fc_layers(x)
        x = self.sigmoid(x) * x1 + self.sigmoid(x1) * x
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    device = "cuda:0"
    model = TOpNet().to(device)
    # 修改测试输入为6通道
    inputs = torch.rand(1, 6, 224, 224).to(device=device)
    out = model(inputs)
    print(out.size())
    total_params = count_parameters(model)
    print(f"Total Parameters: {total_params:,}")
    # summary(model=model, input_size=(6,224,224))
