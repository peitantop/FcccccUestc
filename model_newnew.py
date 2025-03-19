import torch
from torchvision import models
import torch.nn as nn
from model_module import PSConv, FSAS_DFFN, MSCA, HRAMi_DRAMiT, MASAG, DTAB_GCSA, LCA, SSA, EAGFM, BIE_BIEF, SHViTBlock, PCAA
from pdb import set_trace as stop
from FcccccUestc.transformer_layers import SelfAttnLayer
from utils import custom_replace, weights_init
from position_enc import PositionEmbeddingSine, positionalencoding2d
import numpy as np


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
PCAA_block = PCAA.PCAA(512)


# 特征融合
EAGFM_block = EAGFM.EAGFM(1024)
BIEF_block = BIE_BIEF.BIEF(1024)
SHViT_Block = SHViTBlock.SHViTBlock(1024,type='s')
MSCA = MSCA.MSCAttention(1024)
model_base = models.resnet152(weights=models.ResNet152_Weights)
model_base_2 = models.densenet161(weights=models.DenseNet161_Weights)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

class AlternativeNet(nn.Module):
    def __init__(self):
        super(AlternativeNet, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(2208, 1024, kernel_size=1, bias=False)
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
            PCAA_block,
            model_base.layer3,
            )     # -->[1, 1024, 14, 14]
        self.EAGFM = EAGFM_block
        self.strength_block = nn.Sequential(
            SSA_block,  # 注意力模块
            PSConv1,
            PSConv2,  # 风车状卷积
            MSCA,  # 医学图像多尺度交叉轴注意力模块
            model_base.layer4
        )
        self.fc_layers = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.Dropout(0.5),
            nn.Linear(256, 8),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        x4 = self.eff(x)     # efficientnet
        x = self.prex(x)     # resnet -->[1, 64, 56, 56]
        x = self.IEL(x)      # imagestr
        x1 = self.FSASDFFN(x)# imagestr
        x2 = self.DRAMiT(x)  # imagestr
        x = self.MASAG(x1, x2)# iamgestr
        x = self.DTAB(x)     # attention
        x3 = self.conv(x)    # resnet
        x4 = self.AlternativeNet(x4)
        x = self.EAGFM(x3, x4)# eff + res
        x = self.SHViT_Block(x)# featcombine
        x = self.strength_block(x) # res
        return x
    

class CTranModel(nn.Module):
    def __init__(self,num_labels,use_lmt,device, backbone_model, pos_emb=False,layers=3,heads=4,dropout=0.1, int_loss=0, no_x_features=False, grad_cam=False):
        super(CTranModel, self).__init__()
        self.use_lmt = use_lmt
        
        self.no_x_features = no_x_features # (for no image features)

        self.backbone = models.densenet161(weights=models.DenseNet161_Weights).features
        hidden = 0 # this should match the backbone output feature size
        print(backbone_model)

        hidden = 2208

        self.downsample = False
        if self.downsample:
            self.conv_downsample = torch.nn.Conv2d(hidden,hidden,(1,1))
        
        # Label Embeddings
        self.label_input = torch.Tensor(np.arange(num_labels)).view(1,-1).long()
        self.label_lt = torch.nn.Embedding(num_labels, hidden, padding_idx=None)

        # State Embeddings
        self.known_label_lt = torch.nn.Embedding(3, hidden, padding_idx=0)

        # Position Embeddings (for image features)
        self.use_pos_enc = pos_emb
        if self.use_pos_enc:
            # self.position_encoding = PositionEmbeddingSine(int(hidden/2), normalize=True)
            self.position_encoding = positionalencoding2d(hidden, 18, 18).unsqueeze(0)

        # Transformer
        self.self_attn_layers = nn.ModuleList([SelfAttnLayer(hidden,heads,dropout) for _ in range(layers)])

        # Classifier
        # Output is of size num_labels because we want a separate classifier for each label
        self.output_linear = torch.nn.Linear(hidden, num_labels)

        # Other
        self.LayerNorm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        self.grad_cam = grad_cam
        self.hidden = hidden

        # Init all except pretrained backbone
        self.label_lt.apply(weights_init)
        self.known_label_lt.apply(weights_init)
        self.LayerNorm.apply(weights_init)
        self.self_attn_layers.apply(weights_init)
        self.output_linear.apply(weights_init)

        #device
        self.device = device

    def forward(self,images,mask=None):
        const_label_input = self.label_input.repeat(images.size(0),1).to(self.device)
        init_label_embeddings = self.label_lt(const_label_input)

        features = self.backbone(images)
        #print('features', features.shape)
        
        if self.downsample:
            features = self.conv_downsample(features)
        if self.use_pos_enc:
            pos_encoding = self.position_encoding(features,torch.zeros(features.size(0), 18, 18, dtype=torch.bool).to(self.device))
            features = features + pos_encoding

        features = features.view(features.size(0),features.size(1),-1).permute(0,2,1)
        #print('features after weird stuff', features.shape)

        if self.use_lmt:
            # Convert mask values to positive integers for nn.Embedding
            label_feat_vec = custom_replace(mask,0,1,2).long()
            torch.set_printoptions(threshold=10_000)
            # print('original mask: ', mask)
            # print('replaced mask: ', label_feat_vec)

            # Get state embeddings
            state_embeddings = self.known_label_lt(label_feat_vec)
            # print('state_embeddings:', state_embeddings[0, 0])
            # print('label_embeddings:', init_label_embeddings[0, 0])

            # Add state embeddings to label embeddings
            init_label_embeddings += state_embeddings

            # print('masked label_embeddings', init_label_embeddings[0, 0])
        
        if self.no_x_features:
            embeddings = init_label_embeddings 
        else:
            # Concat image and label embeddings
            embeddings = torch.cat((features,init_label_embeddings),1)

        #print('embeddings', embeddings.shape)
        # Feed image and label embeddings through Transformer
        embeddings = self.LayerNorm(embeddings)        
        attns = []
        for layer in self.self_attn_layers:
            embeddings, attn = layer(embeddings, mask=None)
            attns += attn.detach().unsqueeze(0).data

        # Readout each label embedding using a linear layer
        label_embeddings = embeddings[:,-init_label_embeddings.size(1):,:]
        #print('forward label embeddings', label_embeddings.shape)
        output = self.output_linear(label_embeddings)
        #print('forward output linear', output.shape)
        diag_mask = torch.eye(output.size(1)).unsqueeze(0).repeat(output.size(0),1,1).to(self.device)
        #print('diag mask', diag_mask.shape)
        #print(output*diag_mask)
        output = (output*diag_mask).sum(-1)
        #print(output)
        #print('forward output shape:', output.shape)

        if self.grad_cam:
            return output

        # return output, None, attns
        return output


if __name__ == "__main__":
    device = "cuda:0"
    backbone = models.densenet161(weights=models.DenseNet161_Weights).features
    model = CTranModel(num_labels=8,device=device, backbone_model=backbone, use_lmt=None).to(device=device)
    inputs = torch.rand(1, 3, 384, 384).to(device)
    out = model(inputs)
    print(out)
    print(out[0].size())
    # print(model)
    total_params = count_parameters(model)
    print(f"Total Parameters: {total_params:,}")
