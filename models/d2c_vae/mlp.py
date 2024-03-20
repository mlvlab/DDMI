import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
from torch import distributions as dist

from models.d2c_vae.blocks import ToRGB, StyledResBlock, SinusoidalPosEmb, ResnetBlockFC
from utils.general_utils import normalize_coordinate, singleplane_positional_encoding, triplane_positional_encoding, sample_plane_feature

class MLP(nn.Module):
    def __init__(self, *, in_ch=2, latent_dim = 64, out_ch=3, ch=256):
        super().__init__()
        self.latent_dim = latent_dim

        ## Scale-aware Injetion Layers
        dim = int(ch // 4)
        sinu_pos_emb = SinusoidalPosEmb(dim)
        self.time_mlp = nn.Sequential(
                sinu_pos_emb,
                nn.Linear(dim, ch),
                nn.GELU(),
                nn.Linear(ch, ch)
                )

        ## MLP layers
        self.net_res1 = StyledResBlock(in_ch+ latent_dim, ch, 1, ch, demodulate = True, activation = None)
        self.net_res2 = StyledResBlock(ch + in_ch + latent_dim, ch, 1, ch, demodulate = True, activation = None)
        self.net_res3 = StyledResBlock(ch + in_ch + latent_dim, ch, 1, ch, demodulate = True, activation = None)
        self.net_res4 = StyledResBlock(ch, ch, 1, ch, demodulate = True, activation = None)
        self.torgb = ToRGB(ch, out_ch, ch, upsample = False)
       
    def forward(self, coords, hdbf, si):
        # Enables us to compute gradients w.r.t. coordinates
        coords = coords.clone().detach().requires_grad_(True)
        device = coords.device

        _, c, h, w = coords.shape
        b = hdbf[0].shape[0]
        
        assert hdbf is not None and len(hdbf) == 3 # Now, only supports three decomposed BFs.
        coords = coords.repeat(b, 1, 1, 1)
        scale_inj_pixel = torch.ones_like(coords) * si
        coords = coords.permute(0, 2, 3, 1).contiguous()
        scale_inj = torch.ones((b,), device = device) * si
        style = self.time_mlp(scale_inj)
        
        ## Coarse
        x = singleplane_positional_encoding(hdbf[0], coords)
        x = torch.cat((x, scale_inj_pixel), dim = 1)
        ## Middle
        x_m = singleplane_positional_encoding(hdbf[1], coords)
        x_m = torch.cat((x_m, scale_inj_pixel), dim = 1)
        ## Fine
        x_h = singleplane_positional_encoding(hdbf[2], coords)
        x_h = torch.cat((x_h, scale_inj_pixel), dim = 1)

        x = self.net_res1(x, style)
        x = torch.cat((x, x_m), dim = 1) # Concatenation
        x = self.net_res2(x, style)
        x = torch.cat((x, x_h), dim = 1)
        x = self.net_res3(x, style)
        x = self.net_res4(x, style)
        x = self.torgb(x, style)
        return x


class MLP3D(nn.Module):
    def __init__(self, *, in_ch, latent_dim, out_ch, ch=256):
        super().__init__()
        self.latent_dim = latent_dim

        ## MLP layers
        self.net_p = nn.Linear(in_ch, ch)
        self.net_res1 = ResnetBlockFC(latent_dim, ch)
        self.net_res2 = ResnetBlockFC(ch + latent_dim, ch)
        self.net_res3 = ResnetBlockFC(ch + latent_dim, ch)
        self.net_res4 = ResnetBlockFC(ch, ch)
        self.net_out = nn.Linear(ch, out_ch)

    def forward(self, coords, hdbf):
        assert len(hdbf) == 3 # Check tri-plane
        xy_hdbf = hdbf[0]
        yz_hdbf = hdbf[1]
        xz_hdbf = hdbf[2]
        assert len(xy_hdbf) == 3 and len(yz_hdbf) == 3 and len(xz_hdbf) == 3 # Now, only supports three decomposed BFs

        xy_coords = sample_plane_feature(coords, 'xy')
        yz_coords = sample_plane_feature(coords, 'yz')
        xz_coords = sample_plane_feature(coords, 'xz')

        ## Coarse
        x = triplane_positional_encoding(xy_hdbf[0], yz_hdbf[0], xz_hdbf[0], xy_coords, yz_coords, xz_coords)
        x = x.transpose(1, 2)
        ## Middle
        x_m = triplane_positional_encoding(xy_hdbf[1], yz_hdbf[1], xz_hdbf[1], xy_coords, yz_coords, xz_coords)
        x_m = x_m.transpose(1, 2)
        ## Fine
        x_h = triplane_positional_encoding(xy_hdbf[2], yz_hdbf[2], xz_hdbf[2], xy_coords, yz_coords, xz_coords)
        x_h = x_h.transpose(1, 2)

        coords = self.net_p(coords)
        x = coords + self.net_res1(x)
        x = torch.cat((x, x_m), dim = -1)
        x = self.net_res2(x)
        x = torch.cat((x, x_h), dim = -1)
        x = self.net_res3(x)
        x = self.net_res4(x)
        x = self.net_out(x)
        return dist.Bernoulli(logits = x.squeeze(-1))


class MLPVideo(nn.Module):
    def __init__(self, *, in_ch, latent_dim, out_ch, ch=256, **ignore_kwargs):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_ch = out_ch

        ## MLP layers
        self.net_res1 = ResnetBlockFC(latent_dim*3, ch)
        self.net_res2 = ResnetBlockFC(ch + latent_dim*3, ch)
        self.net_res3 = ResnetBlockFC(ch + latent_dim*3, ch)
        self.net_res4 = ResnetBlockFC(ch)
        self.net_out = nn.Linear(ch, out_ch)
        self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, coords, hdbf):
        assert len(hdbf) == 3 # Check tri-plane
        xy_hdbf = hdbf[0]
        yt_hdbf = hdbf[1]
        xt_hdbf = hdbf[2]
        assert len(xy_hdbf) == 3 and len(yt_hdbf) == 3 and len(xt_hdbf) == 3 # Now, only supports three decomposed BFs
        
        b, _, h, w = xy_hdbf[-1].shape
        _, _, t, _ = yt_hdbf[-1].shape
        xy_coords = coords['xy'].repeat(b, 1, 1, 1).permute(0, 2, 3, 1).contiguous()
        yt_coords = coords['yt'].repeat(b, 1, 1, 1).permute(0, 2, 3, 1).contiguous()
        xt_coords = coords['xt'].repeat(b, 1, 1, 1).permute(0, 2, 3, 1).contiguous()

        ## Coarse
        x = triplane_positional_encoding(xy_hdbf[0], yt_hdbf[0], xt_hdbf[0], xy_coords, yt_coords, xt_coords, mode = 'concat')
        ## Middle
        x_m = triplane_positional_encoding(xy_hdbf[1], yt_hdbf[1], xt_hdbf[1], xy_coords, yt_coords, xt_coords, mode = 'concat')
        ## Fine
        x_h = triplane_positional_encoding(xy_hdbf[2], yt_hdbf[2], xt_hdbf[2], xy_coords, yt_coords, xt_coords, mode = 'concat')

        x = self.net_res1(x)
        x = torch.cat((x, x_m), dim = 1)
        x = self.net_res2(x)
        x = torch.cat((x, x_h), dim = 1)
        x = self.net_res3(x)
        x = self.net_res4(x)
        x = self.net_out(self.actvn(x))
        x = x.reshape(b, -1, self.out_ch)
        x = x.permute(0, 2, 1).reshape(b, self.out_ch, t, h, w)
        return x

