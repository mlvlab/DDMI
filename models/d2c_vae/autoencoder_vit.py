"""
wild mixture and modification of
https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/diffusionmodules/model.py
https://github.com/sihyun-yu/PVDM/blob/main/models/autoencoder/autoencoder_vit.py
"""

import math
import numpy as np
from functools import partial
from collections import namedtuple

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from PIL import Image
from models.d2c_vae.vit_modules import TimeSformerEncoder
from models.d2c_vae.autoencoder_unet import Decoder, VideoDecoder, VideoDecoder_light
from models.ldm.modules.distributions import DiagonalGaussianDistribution

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

# ====================================================================================================

class VITAutoencoder(nn.Module):
    def __init__(
        self,
        ddconfig,
        embed_dim,
        frames,
    ):
        super().__init__()
        
        self.res = ddconfig['resolution']
        self.embed_dim = embed_dim
        self.timesformer_channels = ddconfig['timesformer_channels']
        self.splits = ddconfig['splits']
        self.frames = frames // self.splits
        self.double_z = ddconfig['double_z']
        self.z_channels = ddconfig['z_channels']
        self.downsample_factor = 3

        patch_size = ddconfig['patch_size']
        if self.res == 128:
            patch_size = 4

        ## Encoder
        self.encoder = TimeSformerEncoder(
            dim = self.timesformer_channels,
            image_size = self.res,
            num_frames = self.frames,
            depth = 8,
            patch_size = patch_size,
        )

        self.xy_token = nn.Parameter(torch.randn(1, 1, self.timesformer_channels))
        self.xt_token = nn.Parameter(torch.randn(1, 1, self.timesformer_channels))
        self.yt_token = nn.Parameter(torch.randn(1, 1, self.timesformer_channels))

        self.xy_pos_embedding = nn.Parameter(torch.randn(1, self.frames + 1, self.timesformer_channels))
        self.xt_pos_embedding = nn.Parameter(torch.randn(1, self.res // (2**self.downsample_factor) + 1, self.timesformer_channels))
        self.yt_pos_embedding = nn.Parameter(torch.randn(1, self.res // (2**self.downsample_factor) + 1, self.timesformer_channels))
        
        self.xy_quant_attn = Transformer(self.timesformer_channels, 4, 4, self.timesformer_channels // 8, 512)
        self.yt_quant_attn = Transformer(self.timesformer_channels, 4, 4, self.timesformer_channels // 8, 512)
        self.xt_quant_attn = Transformer(self.timesformer_channels, 4, 4, self.timesformer_channels // 8, 512)

        self.pre_xy = torch.nn.Conv2d(self.timesformer_channels, 2*self.embed_dim if self.double_z else self.embed_dim, 1)
        self.pre_xt = torch.nn.Conv2d(self.timesformer_channels, 2*self.embed_dim if self.double_z else self.embed_dim, 1)
        self.pre_yt = torch.nn.Conv2d(self.timesformer_channels, 2*self.embed_dim if self.double_z else self.embed_dim, 1)


        ## Decoder
        self.post_xy = torch.nn.Conv2d(self.embed_dim, self.z_channels, 1)
        self.post_xt = torch.nn.Conv2d(self.embed_dim, self.z_channels, 1)
        self.post_yt = torch.nn.Conv2d(self.embed_dim, self.z_channels, 1)

        self.decoder = VideoDecoder_light(**ddconfig)

    def encode(self, x):
        b = x.size(0)
        x = rearrange(x, 'b c t h w -> b t c h w')
        x = self.encoder(x)
        x = rearrange(x, 'b (t h w) c -> b c t h w', t= self.frames, h=self.res // (2**self.downsample_factor))

        xy = rearrange(x, 'b c t h w -> (b h w) t c')
        n = xy.size(1)
        xy_token = repeat(self.xy_token, '1 1 d -> bhw 1 d', bhw = xy.size(0))
        xy = torch.cat((xy, xy_token), dim = 1)
        xy += self.xy_pos_embedding[:,:(n+1)]
        xy= self.xy_quant_attn(xy)[:,0]
        xy = rearrange(xy, '(b h w) c -> b c h w', b=b, h=self.res // (2**self.downsample_factor))

        yt = rearrange(x, 'b c t h w -> (b t w) h c')
        n = yt.size(1)
        yt_token = repeat(self.yt_token, '1 1 d -> bth 1 d', bth = yt.size(0))
        yt = torch.cat((yt, yt_token), dim = 1)
        yt += self.yt_pos_embedding[:,:(n+1)]
        yt = self.yt_quant_attn(yt)[:,0]
        yt = rearrange(yt, '(b t h) c -> b c t h', b=b, h=self.res // (2**self.downsample_factor))

        xt = rearrange(x, 'b c t h w -> (b t h) w c')
        n=xt.size(1)
        xt_token = repeat(self.xt_token, '1 1 d -> btw 1 d', btw = xt.size(0))
        xt = torch.cat((xt, xt_token), dim = 1)
        xt += self.xt_pos_embedding[:,:(n+1)]
        xt = self.xt_quant_attn(xt)[:,0]
        xt = rearrange(xt, '(b t w) c -> b c t w', b=b, w=self.res // (2**self.downsample_factor))

        xy_latent = self.pre_xy(xy)
        yt_latent = self.pre_yt(yt)
        xt_latent = self.pre_xt(xt)
        
        xy_posterior = DiagonalGaussianDistribution(xy_latent)
        yt_posterior = DiagonalGaussianDistribution(yt_latent)
        xt_posterior = DiagonalGaussianDistribution(xt_latent)

        return xy_posterior, yt_posterior, xt_posterior

    def decode(self, x):
        size1=self.res // (2**self.downsample_factor)
        size2=self.res // (2**self.downsample_factor)
        size3 = self.frames
        xy = x[:, :, 0:size1*size2].view(x.size(0), x.size(1), size1, size2)
        xt = x[:, :, size1*size2:size1*(size2+size3)].view(x.size(0), x.size(1), size3, size2)
        yt = x[:, :, size1*(size2+size3):size1*(size2+size3+size3)].view(x.size(0), x.size(1), size3, size2)

        xy = self.post_xy(xy)
        yt = self.post_yt(yt)
        xt = self.post_xt(xt)

        xy, yt, xt = self.decoder((xy, yt, xt))
        
        return xy, yt, xt

    def forward(self, x, sample_posterior=True):
        xy_posterior, yt_posterior, xt_posterior = self.encode(x)

        if sample_posterior:
            xy = xy_posterior.sample()
            yt = yt_posterior.sample()
            xt = xt_posterior.sample()
        else:
            xy = xy_posterior.mode()
            yt = yt_posterior.mode()
            xt = xt_posterior.mode()
        
        b, c = xy.shape[0], xy.shape[1]
        x = torch.cat([xy.reshape(b, c, -1), xt.reshape(b, c, -1), yt.reshape(b, c, -1)], dim = 2)
        dec = self.decode(x)
        
        return dec, xy_posterior, yt_posterior, xt_posterior


