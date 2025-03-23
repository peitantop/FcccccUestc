import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, t):
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        emb = t[:, None] * emb[None, :].to(t.device)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return self.proj(emb)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.time_mlp = nn.Linear(time_dim, out_channels)

    def forward(self, x, t):
        h = F.silu(self.norm1(self.conv1(x)))
        time_emb = F.silu(self.time_mlp(t))
        h = h + time_emb[..., None, None]
        h = F.silu(self.norm2(self.conv2(h)))
        return h

class EyeDDPM(nn.Module):
    def __init__(self, image_size=224, channels=3, time_dim=256):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = TimeEmbedding(time_dim)
        
        # U-Net架构
        self.down1 = ConvBlock(channels, 64, time_dim)
        self.down2 = ConvBlock(64, 128, time_dim)
        self.down3 = ConvBlock(128, 256, time_dim)
        self.down4 = ConvBlock(256, 512, time_dim)
        
        self.up1 = ConvBlock(512+256, 256, time_dim)
        self.up2 = ConvBlock(256+128, 128, time_dim)
        self.up3 = ConvBlock(128+64, 64, time_dim)
        self.final = nn.Conv2d(64, channels, 1)
        
        self.downs = nn.ModuleList([
            nn.MaxPool2d(2) for _ in range(3)
        ])
        self.ups = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode='bilinear') for _ in range(3)
        ])

    def forward(self, x, t):
        t = self.time_mlp(t)
        
        h1 = self.down1(x, t)
        x = self.downs[0](h1)
        h2 = self.down2(x, t)
        x = self.downs[1](h2)
        h3 = self.down3(x, t)
        x = self.downs[2](h3)
        x = self.down4(x, t)
        
        x = self.ups[0](x)
        x = torch.cat([x, h3], dim=1)
        x = self.up1(x, t)
        x = self.ups[1](x)
        x = torch.cat([x, h2], dim=1)
        x = self.up2(x, t)
        x = self.ups[2](x)
        x = torch.cat([x, h1], dim=1)
        x = self.up3(x, t)
        
        return self.final(x)

def generate_samples(model, n_samples, device, n_steps=1000):
    """生成新的眼底图像样本"""
    model.eval()
    with torch.no_grad():
        # 从随机噪声开始
        x = torch.randn(n_samples, 3, 224, 224).to(device)
        
        # 逐步去噪
        for t in tqdm(range(n_steps-1, -1, -1)):
            time_tensor = torch.ones(n_samples).to(device) * t
            predicted_noise = model(x, time_tensor)
            alpha_t = 1 - (t / n_steps)
            x = (x - (1 - alpha_t) * predicted_noise) / alpha_t.sqrt()
            if t > 0:
                noise = torch.randn_like(x)
                x = x + noise * (t / n_steps).sqrt()
    
    return x.cpu()