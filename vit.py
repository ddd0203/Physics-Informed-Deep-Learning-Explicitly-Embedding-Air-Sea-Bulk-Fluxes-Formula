"""
This file is adapted from Nguyen, Tung, et al. "ClimaX: A foundation model
for weather and climate." arXiv preprint arXiv:2301.10343 (2023).
Code from this project is available at https://github.com/microsoft/ClimaX
"""

from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block, PatchEmbed, trunc_normal_

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
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


def interpolate_pos_embed(model, checkpoint_model, new_size=(64, 128)):
    if "net.pos_embed" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["net.pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        orig_num_patches = pos_embed_checkpoint.shape[-2]
        patch_size = model.patch_size
        w_h_ratio = 2
        orig_h = int((orig_num_patches // w_h_ratio) ** 0.5)
        orig_w = w_h_ratio * orig_h
        orig_size = (orig_h, orig_w)
        new_size = (new_size[0] // patch_size, new_size[1] // patch_size)

        if orig_size[0] != new_size[0]:
            print(
                "Interpolate PEs from %dx%d to %dx%d"
                % (orig_size[0], orig_size[1], new_size[0], new_size[1])
            )
            pos_tokens = pos_embed_checkpoint.reshape(
                -1, orig_size[0], orig_size[1], embedding_size
            ).permute(0, 3, 1, 2)
            new_pos_tokens = torch.nn.functional.interpolate(
                pos_tokens,
                size=(new_size[0], new_size[1]),
                mode="bicubic",
                align_corners=False,
            )
            new_pos_tokens = new_pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            checkpoint_model["net.pos_embed"] = new_pos_tokens


def interpolate_channel_embed(checkpoint_model, new_len):
    if "net.channel_embed" in checkpoint_model:
        channel_embed_checkpoint = checkpoint_model["net.channel_embed"]
        old_len = channel_embed_checkpoint.shape[1]
        if new_len <= old_len:
            checkpoint_model["net.channel_embed"] = channel_embed_checkpoint[
                :, :new_len
            ]
'''
class InterPatchMixer(nn.Module):
    """
    插入到 PatchEmbed 之后。
    将 (B, N, D) 还原为 2D 网格，进行卷积融合，再展平。
    作用：让 Transformer 看到的 Token 包含邻域信息，而不仅仅是孤立的切片。
    """
    def __init__(self, embed_dim, img_size, patch_size):
        super().__init__()
        self.H_grid = img_size[0] // patch_size
        self.W_grid = img_size[1] // patch_size
        
        self.mixer = nn.Sequential(
            # 1. 深度卷积 (Depthwise): 独立处理每个通道，计算空间梯度
            # padding=1 + circular 模式对于全球气象数据非常重要
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim, padding_mode='circular'),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            
            # 2. 点卷积 (Pointwise): 通道间混合
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
            nn.GELU()
        )

    def forward(self, x):
        # x shape: (B, N, D)
        B, N, D = x.shape
        
        # 1. 还原为 2D 空间结构: (B, D, H_grid, W_grid)
        x = x.transpose(1, 2).reshape(B, D, self.H_grid, self.W_grid)
        
        # 2. 执行卷积融合 (捕捉 Patch 间的梯度)
        x = self.mixer(x)
        
        # 3. 展平回 Token 序列: (B, N, D)
        x = x.flatten(2).transpose(1, 2)
        return x
'''

class CNNPredictionHead(nn.Module):
    def __init__(self, embed_dim, out_channels, patch_size, img_size):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # 计算 Grid 大小 (例如 256/8 = 32)
        self.H_grid = img_size[0] // patch_size
        self.W_grid = img_size[1] // patch_size

        # 1. 特征融合层：使用 3x3 卷积，padding=1。
        # 核心作用：让 grid (x, y) 能看到 (x-1, y) 和 (x+1, y) 的信息，消除边界断层。
        self.smooth_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(embed_dim) # 或者 LayerNorm
        )

        # 2. 上采样层：这里使用 PixelShuffle (高效) 或 ConvTranspose2d (转置卷积)
        # 这一步把 (B, D, H/P, W/P) -> (B, Out, H, W)
        self.upsample = nn.PixelShuffle(patch_size)
        
        # PixelShuffle 会把通道数除以 factor^2，所以我们需要先调整通道数
        # 输入通道: embed_dim
        # 需要输出的通道(PixelShuffle前): out_channels * (patch_size ** 2)
        self.proj = nn.Conv2d(embed_dim, out_channels * (patch_size ** 2), kernel_size=1)

    def forward(self, x):
        # x shape: (B, Num_Patches, Embed_Dim)
        B, N, D = x.shape
        
        # 1. 恢复成 2D 特征图: (B, Embed_Dim, H_grid, W_grid)
        x = x.transpose(1, 2).reshape(B, D, self.H_grid, self.W_grid)
        
        # 2. 执行平滑卷积 (这是解决马赛克的关键步骤！)
        x = self.smooth_conv(x)
        
        # 3. 投影通道
        x = self.proj(x)
        
        # 4. 上采样变回原图分辨率: (B, Out_Channels, H, W)
        x = self.upsample(x)
        
        return x


class ViT(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        h_channels,
        img_size=[256, 128],
        patch_size=8,
        depth=24,
        decoder_depth=4,
        num_heads=16,
        mlp_ratio=4.0,
        drop_path=0.0,
        drop_rate=0.0,
        per_var_embedding=True,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        default_vars = [str(i) for i in range(in_channels)]
        self.default_vars = default_vars
        embed_dim = h_channels
        self.per_var_embedding = per_var_embedding

        if self.per_var_embedding:
            self.token_embeds = nn.ModuleList(
                [
                    PatchEmbed(img_size, patch_size, 1, embed_dim)
                    for i in range(len(default_vars))
                ]
            )
        else:
            self.token_embeds = nn.ModuleList(
                [PatchEmbed(img_size, patch_size, in_channels, embed_dim)]
            )
        self.num_patches = self.token_embeds[0].num_patches

        self.var_embed, self.var_map = self.create_var_embedding(embed_dim)
        self.var_query = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.var_agg = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim), requires_grad=True
        )
        self.lead_time_embed = nn.Linear(1, embed_dim)

        self.out_dim = out_channels
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    drop_path=dpr[i],
                    norm_layer=nn.LayerNorm,
                    #drop=drop_rate,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        #self.head = nn.ModuleList()
        #for _ in range(decoder_depth):
        #    self.head.append(nn.Linear(embed_dim, embed_dim))
        #    self.head.append(nn.GELU())
        #self.head.append(nn.Linear(embed_dim, self.out_dim * patch_size**2))
        #self.head = nn.Sequential(*self.head)

        self.pre_head = nn.ModuleList()
        for _ in range(decoder_depth):
            self.pre_head.append(nn.Linear(embed_dim, embed_dim))
            self.pre_head.append(nn.GELU())
        self.pre_head = nn.Sequential(*self.pre_head)

        #self.patch_mixer = InterPatchMixer(
        #    embed_dim=h_channels, 
        #    img_size=img_size, 
        #    patch_size=patch_size
        #)

        # 使用我们定义的新 CNNHead 替换最后的 Linear
        self.head = CNNPredictionHead(
            embed_dim=embed_dim, 
            out_channels=self.out_dim, 
            patch_size=patch_size, 
            img_size=img_size
        )

        self.initialize_weights()
        if not self.per_var_embedding:
            #self.mlp = MLP(in_channels=14, out_channels=14)#noera5
            #self.mlp = MLP(in_channels=26, out_channels=26)#era5_raw
            self.mlp = MLP(in_channels=38, out_channels=38)#era5_flux

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.img_size[0] / self.patch_size),
            int(self.img_size[1] / self.patch_size),
            cls_token=False,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        var_embed = get_1d_sincos_pos_embed_from_grid(
            self.var_embed.shape[-1], np.arange(len(self.default_vars))
        )
        self.var_embed.data.copy_(torch.from_numpy(var_embed).float().unsqueeze(0))

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

    def create_var_embedding(self, dim):
        var_embed = nn.Parameter(
            torch.zeros(1, len(self.default_vars), dim), requires_grad=True
        )
        var_map = {}
        idx = 0
        for var in self.default_vars:
            var_map[var] = idx
            idx += 1
        return var_embed, var_map

    @lru_cache(maxsize=None)
    def get_var_ids(self, vars, device):
        ids = np.array([self.var_map[var] for var in vars])
        return torch.from_numpy(ids).to(device)

    def get_var_emb(self, var_emb, vars):
        ids = self.get_var_ids(vars, var_emb.device)
        return var_emb[:, ids, :]

    def unpatchify(self, x: torch.Tensor, h=None, w=None):
        p = self.patch_size
        c = self.out_dim
        h = self.img_size[0] // p if h is None else h // p
        w = self.img_size[1] // p if w is None else w // p
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def aggregate_variables(self, x: torch.Tensor):
        b, _, l, _ = x.shape
        x = torch.einsum("bvld->blvd", x)
        x = x.flatten(0, 1)

        var_query = self.var_query.repeat_interleave(x.shape[0], dim=0)
        x, _ = self.var_agg(var_query, x, x)
        x = x.squeeze()

        x = x.unflatten(dim=0, sizes=(b, l))
        return x

    def mlp_embedding(self, x):

        return

    def forward_encoder(self, x, lead_times, variables):

        if isinstance(variables, list):
            variables = tuple(variables)

        if self.per_var_embedding:
            embeds = []
            var_ids = self.get_var_ids(variables, x.device)
            for i in range(len(var_ids)):
                id = var_ids[i]
                embeds.append(self.token_embeds[id](x[:, i : i + 1]))
            x = torch.stack(embeds, dim=1)

            var_embed = self.get_var_emb(self.var_embed, variables)
            x = x + var_embed.unsqueeze(2)

            x = self.aggregate_variables(x)

        else:
            x = self.mlp(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            x = self.token_embeds[0](x)
            #x = self.patch_mixer(x)

        x = x + self.pos_embed

        lead_time_emb = self.lead_time_embed(lead_times.unsqueeze(-1))
        lead_time_emb = lead_time_emb.unsqueeze(1)

        x = x + lead_time_emb

        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x
    '''
    def forward(self, x, lead_times=None, film_index=None):
    
        if lead_times is None:
            lead_times = torch.ones(x.shape[0]).float().cuda().unsqueeze(-1)

        out_transformers = self.forward_encoder(x, lead_times[:, 0], self.default_vars)
        preds = self.head(out_transformers)
        preds = self.unpatchify(preds)

        return preds.permute(0, 2, 3, 1)
    '''

    def forward(self, x, lead_times=None, film_index=None):
        if lead_times is None:
            # 注意：如果这里报错设备问题，建议改为 x.device
            lead_times = torch.ones(x.shape[0]).float().to(x.device).unsqueeze(-1)

        # 1. 经过 Encoder 得到 Patch Token
        out_transformers = self.forward_encoder(x, lead_times[:, 0], self.default_vars)
        
        # 2. 先经过原来的 MLP 增加非线性 (对应修改后的 self.pre_head)
        out_transformers = self.pre_head(out_transformers)

        # 3. 经过新的 CNN Head，内部自动 reshape + 平滑 + 上采样
        # 输出形状直接就是 (B, C, H, W)
        preds = self.head(out_transformers) 

        # 4. 调整维度的顺序以匹配原来的输出格式 (B, H, W, C)
        return preds.permute(0, 2, 3, 1)