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


# Nerf

class Embedding(nn.Module):
    def __init__(self, in_channels, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super(Embedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels*(len(self.funcs)*N_freqs+1)

        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...) 
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12

        Inputs:
            x: (B, self.in_channels)

        Outputs:
            out: (B, self.out_channels)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq*x)]

        return torch.cat(out, -1)


class MLPNeRF(nn.Module):
    def __init__(self,
                 D=8, W=256,
                 in_channels_xyz=96, in_channels_dir=27, 
                 skips=[2,4,6]):
        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
        skips: add skip connection in the Dth layer
        """
        super(MLPNeRF, self).__init__()
        self.D = D
        self.W = W
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir
        self.skips = skips

        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, W)
            elif i in skips:
                layer = nn.Linear(W+in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.LeakyReLU(True))
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(W, W)

        # direction encoding layers
        self.dir_encoding = nn.Sequential(
                                nn.Linear(W+in_channels_dir, W//2),
                                nn.LeakyReLU(True))

        # output layers
        self.sigma = nn.Linear(W, 1)
        self.rgb = nn.Sequential(
                        nn.Linear(W//2, 3),
                        nn.Sigmoid())

    def forward(self, x, sigma_only=False):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py

        Inputs:
            x: (B, self.in_channels_xyz(+self.in_channels_dir))
               the embedded vector of position and direction
            sigma_only: whether to infer sigma only. If True,
                        x is of shape (B, self.in_channels_xyz)

        Outputs:
            if sigma_ony:
                sigma: (B, 1) sigma
            else:
                out: (B, 4), rgb and sigma
        """
        if not sigma_only:
            input_xyz, input_dir = \
                torch.split(x, [self.in_channels_xyz, self.in_channels_dir], dim=-1)
        else:
            input_xyz = x

        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        sigma = self.sigma(xyz_)
        if sigma_only:
            return sigma

        xyz_encoding_final = self.xyz_encoding_final(xyz_)

        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir], -1)
        dir_encoding = self.dir_encoding(dir_encoding_input)
        rgb = self.rgb(dir_encoding)

        out = torch.cat([rgb, sigma], -1)

        return out