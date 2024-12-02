from setup import *

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models_mae import MaskedAutoencoderViT
from models_vit import ViT


# slime yandex cup 2022
class AttentionPooling(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(embedding_size, embedding_size), \
            nn.LayerNorm(embedding_size),
            nn.GELU(), 
            nn.Linear(embedding_size, 1)
        )
    def forward(self, x, mask=None):
        attn_logits = self.attn(x)
        if mask is not None:
            attn_logits[mask] = -float('inf')
        attn_weights = torch.softmax(attn_logits, dim=1)
        x = x * attn_weights
        x = x.sum(dim=1)
        return x
    

# https://github.com/psinger/kaggle-landmark-recognition-2020-1st-place/blob/main/src/loss.py
class ArcFaceLoss(nn.modules.Module):
    def __init__(self, s=45.0, m=0.1, crit="bce", weight=None, reduction="mean", class_weights_norm=None):
        super().__init__()

        self.weight = weight
        self.reduction = reduction
        self.class_weights_norm = class_weights_norm
        
        self.crit = nn.CrossEntropyLoss(reduction="none")   
        
        if s is None:
            self.s = torch.nn.Parameter(torch.tensor([45.], requires_grad=True, device='cuda'))
        else:
            self.s = s
        # self.s = torch.nn.Parameter(torch.tensor([45.], requires_grad=True, device='cuda'))
        
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        
    def forward(self, logits, labels):

        logits = logits.float()
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        labels2 = torch.zeros_like(cosine)
        labels2.scatter_(1, labels.view(-1, 1).long(), 1)
        output = (labels2 * phi) + ((1.0 - labels2) * cosine)

        s = self.s

        output = output * s
        loss = self.crit(output, labels)

        if self.weight is not None:
            w = self.weight[labels].to(logits.device)

            loss = loss * w
            if self.class_weights_norm == "batch":
                loss = loss.sum() / w.sum()
            if self.class_weights_norm == "global":
                loss = loss.mean()
            else:
                loss = loss.mean()
            
            return loss

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss
    

# https://github.com/psinger/kaggle-landmark-recognition-2020-1st-place/blob/main/src/loss.py
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        return cosine


class ArcModel(nn.Module):
    def __init__(self, num_classes):
        super(ArcModel, self).__init__()

        self.num_classes = num_classes

        self.mae = MaskedAutoencoderViT(
            img_size=(84, 52),
            patch_size=(4, 4),
            in_chans=1,
            embed_dim=768,
            depth=16,
            decoder_depth=6,
            decoder_embed_dim=384,
            mlp_ratio=4,
            num_heads=12,
            decoder_num_heads=6,
        )

        self.vit = ViT(
            image_size=(84, 52),
            patch_size=(4, 4),
            num_classes=num_classes,
            dim=768,
            depth=3,
            heads=12,
            mlp_dim=768 * 4,
            pool='mean',
            channels=1,
            dim_head=64,
        )

        self.ap = AttentionPooling(768)
        self.bn = nn.BatchNorm1d(768)
        self.proj = nn.Linear(768, 768)
        self.arcface_head = ArcMarginProduct(
            in_features=768,
            out_features=num_classes
        )

    def unfreeze(self, step):
        if step <= 0:
            n_freeze = 16
        elif step <= 1:
            n_freeze = 11
        elif step <= 2:
            n_freeze = 6
        else:
            n_freeze = 0 

        self.mae.patch_embed.requires_grad_(False)
        for i, block in enumerate(self.mae.blocks):
            if i < n_freeze:
                block.requires_grad_(False)
            else:
                block.requires_grad_(True)

        print(f"Freezing {n_freeze} layers")

    def forward(self, x, mask_ratio=0.0):    
        x = F.pad(x, (0, 2, 0, 0), "constant", 0).unsqueeze(1)
        x = (x - MEAN) / STD

        x, mask, ids_restore = self.mae.forward_encoder(x, mask_ratio=mask_ratio)

        x = self.proj(x)
        x = self.vit.transformer(x)
        x = self.ap(x)

        x = self.bn(x)
        x = F.normalize(x)
        
        return x