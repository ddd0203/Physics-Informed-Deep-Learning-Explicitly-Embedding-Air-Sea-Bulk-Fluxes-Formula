"""
This file is adapted from Nguyen, Tung, et al. "ClimaX: A foundation model
for weather and climate." arXiv preprint arXiv:2301.10343 (2023).
Code from this project is available at https://github.com/microsoft/ClimaX

Integrated with Swin-Transformer V2 & Auto-Padding/Cropping mechanism by DouJiaqi.
"""

from functools import lru_cache
import math

import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, trunc_normal_
from timm.models.swin_transformer_v2 import SwinTransformerV2Block

from architectures import MLP

def get_2d_sincos_pos_embed(embed_dim, grid_size_h, grid_size_w, cls_token=False):
    grid_h = np.arange(grid_size_h, dtype=np.float32)
    grid_w = np.arange(grid_size_w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size_h, grid_size_w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb

class CNNPredictionHead(nn.Module):
    def __init__(self, embed_dim, out_channels, patch_size, padded_img_size):
        super().__init__()
        # 使用 Padding 后的新尺寸
        self.H_grid = padded_img_size[0] // patch_size
        self.W_grid = padded_img_size[1] // patch_size
        self.patch_size = patch_size

        self.smooth_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(embed_dim)
        )

        self.upsample = nn.PixelShuffle(patch_size)
        self.proj = nn.Conv2d(embed_dim, out_channels * (patch_size ** 2), kernel_size=1)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        # 2. 执行平滑卷积
        x = self.smooth_conv(x)
        # 3. 投影通道
        x = self.proj(x)
        # 4. 上采样变回原图分辨率倍数
        x = self.upsample(x)
        return x

class ViT(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            h_channels,
            img_size=[241, 397],
            patch_size=4,
            depth=4,
            decoder_depth=8,
            num_heads=16,
            mlp_ratio=4.0,
            drop_path=0.0,
            drop_rate=0.0,
            window_size=7,
    ):
        super().__init__()

        self.ori_img_size = img_size
        self.patch_size = patch_size
        self.window_size = window_size

        # ✅ 核心对齐逻辑：使宽和高能够同时被 patch_size 和 window_size 的乘积整除
        # 乘积因子 (例如 4 * 7 = 28)
        align_factor = patch_size * window_size

        # 计算需要的 Padded Height 和 Padded Width
        padded_h = math.ceil(img_size[0] / align_factor) * align_factor
        padded_w = math.ceil(img_size[1] / align_factor) * align_factor

        self.padded_img_size = [int(padded_h), int(padded_w)]
        self.pad_h_total = self.padded_img_size[0] - img_size[0]
        self.pad_w_total = self.padded_img_size[1] - img_size[1]

        # Swin 内部特征网格大小
        self.grid_h = self.padded_img_size[0] // patch_size
        self.grid_w = self.padded_img_size[1] // patch_size

        default_vars = [str(i) for i in range(in_channels)]
        self.default_vars = default_vars
        embed_dim = h_channels

        # PatchEmbed 需要用 Padded 后的输入尺寸
        self.token_embeds = nn.ModuleList(
                [PatchEmbed(self.padded_img_size, patch_size, in_channels, embed_dim)]
            )
        self.num_patches = self.token_embeds[0].num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim), requires_grad=True
        )
        self.lead_time_embed = nn.Linear(1, embed_dim)

        self.out_dim = out_channels
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]

        self.blocks = nn.ModuleList([
            SwinTransformerV2Block(
                dim=embed_dim,
                input_resolution=(self.grid_h, self.grid_w),
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else (window_size // 2),
                mlp_ratio=mlp_ratio,
                drop_path=dpr[i]
            )
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        self.pre_head = nn.ModuleList()
        for _ in range(decoder_depth):
            self.pre_head.append(nn.Linear(embed_dim, embed_dim))
            self.pre_head.append(nn.GELU())
        self.pre_head = nn.Sequential(*self.pre_head)

        # 传入的 head 需要知道 Padding 后生成的 token size，方便 reshape 还原
        self.head = CNNPredictionHead(
            embed_dim=embed_dim,
            out_channels=self.out_dim,
            patch_size=patch_size,
            padded_img_size=self.padded_img_size
        )

        self.initialize_weights()
        #self.mlp = MLP(in_channels=199, out_channels=199)  # era5_flux
        #self.mlp = MLP(in_channels=30, out_channels=30) #no_futu_flux
        self.mlp = MLP(in_channels=38, out_channels=38) #futu_flux
        #self.mlp = MLP(in_channels=14, out_channels=14) #era5_no
        #self.mlp = MLP(in_channels=35, out_channels=35) #era5_raw
        #self.mlp = MLP(in_channels=26, out_channels=26) #era5_raw_old
        #self.mlp = MLP(in_channels=59, out_channels=59) #era5_flux+raw

    def initialize_weights(self):
        # 使用补齐后的大网格生成绝对位置编码
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            self.grid_h,
            self.grid_w,
            cls_token=False,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        for i in range(len(self.token_embeds)):
            w = self.token_embeds[i].proj.weight.data
            trunc_normal_(w.view([w.shape[0], -1]), std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, x, lead_times, variables):
        if isinstance(variables, list):
            variables = tuple(variables)

        # 经过 mlp 和 Patch 嵌入
        x = self.mlp(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.token_embeds[0](x)
        x = x + self.pos_embed

        lead_time_emb = self.lead_time_embed(lead_times.unsqueeze(-1))
        lead_time_emb = lead_time_emb.unsqueeze(1)
        x = x + lead_time_emb
        x = self.pos_drop(x)
        B, N, C = x.shape
        x = x.view(B, self.grid_h, self.grid_w, C)

        # 逐层经过 SwinV2 Block
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def forward(self, x, lead_times=None, film_index=None):
        if self.pad_h_total > 0 or self.pad_w_total > 0:
            # (pad_left, pad_right, pad_top, pad_bottom)
            x = nn.functional.pad(x, (0, self.pad_w_total, 0, self.pad_h_total), mode='replicate')

        if lead_times is None:
            lead_times = torch.ones(x.shape[0]).float().to(x.device).unsqueeze(-1)
        out_transformers = self.forward_encoder(x, lead_times[:, 0], self.default_vars)
        out_transformers = self.pre_head(out_transformers)
        preds = self.head(out_transformers)
        preds = preds[:, :, :self.ori_img_size[0], :self.ori_img_size[1]]
        return preds.permute(0, 2, 3, 1)