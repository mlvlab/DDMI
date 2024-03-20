import pdb
import math
import numpy as np
from functools import partial
from collections import namedtuple

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange
from PIL import Image

# constants
ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

def l2norm(t):
    return F.normalize(t, dim = -1)

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# small helper modules

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class LearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with learned sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        if 0:
            self.mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, dim_out * 2)
            ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if 0:
            if exists(self.mlp) and exists(time_emb):
                time_emb = self.mlp(time_emb)
                time_emb = rearrange(time_emb, 'b c -> b c 1 1')
                scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32, scale = 16):
        super().__init__()
        self.scale = scale
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q, k = map(l2norm, (q, k))

        sim = einsum('b h d i, b h d j -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h d j -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

class ChannelAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32, scale = 16):
        super().__init__()
        self.scale = scale
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h = self.heads), qkv)

        q, k = map(l2norm, (q, k))

        sim = einsum('b h d i, b h d j -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h d j -> b h i d', attn, v)
        out = rearrange(out, 'b h d (x y)-> b (h d) x y', x = h, y = w)
        return self.to_out(out)


# model

class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        dim_mults=(1, 1, 2, 2, 4, 4),
        channels = 3,
        resnet_block_groups = 8,
        input_size = 32,
        out_channels = 3,
        downsample_factor=3,
        upsample_factor=2
    ):
        super().__init__()

        # determine dimensions
        self.input_size = input_size
        self.channels = channels
        self.downsample_factor = downsample_factor
        self.upsample_factor = upsample_factor
        input_channels = channels

        init_dim = default(init_dim, dim)
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # layers
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # Downsampling
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_downsample = ind < self.downsample_factor
            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in),
                block_klass(dim_in, dim_in),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if is_downsample else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_dim = mid_dim
        self.mid_block1 = block_klass(mid_dim, mid_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim)

        self.mid_block3 = block_klass(mid_dim, mid_dim)
        self.mid_attn2 = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block4 = block_klass(mid_dim, mid_dim)

        self.normalize = nn.GroupNorm(16, mid_dim)
        self.nonlinear = nn.SiLU()
        self.conv_enc = nn.Conv2d(mid_dim, mid_dim * 2, 3, padding = 1)


        # Upsampling
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            idx_up = len(in_out) - 1 - self.upsample_factor
            is_upsample = ind >= idx_up and ind <= (idx_up + self.upsample_factor - 1)

            idx_pe = len(in_out) - 2
            #is_last_enc = ind == idx_pe or ind == (idx_pe - 1)
            is_last_enc = 0

            self.ups.append(nn.ModuleList([
                block_klass(dim_out, dim_out),
                block_klass(dim_out, dim_out),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                nn.Conv2d(dim_out, out_channels, 1) if is_last_enc else nn.Identity(),
                Upsample(dim_out, dim_in) if is_upsample else  nn.Conv2d(dim_out, dim_in, 3, padding = 1),
            ]))

        self.final_res_block = block_klass(dim, dim)
        self.final_conv = nn.Conv2d(dim, out_channels, 1)

    def forward(self, x = None, latent = None):
        lists = []
        if latent is None:
            x = self.init_conv(x)
            r = x.clone()
            h = []
            for block1, block2, attn, downsample in self.downs:
                x = block1(x)
                x = block2(x)
                x = attn(x)
                x = downsample(x)
            
            x = self.mid_block1(x)
            x = self.mid_attn(x)
            x = self.mid_block2(x)
            x = self.normalize(x)
            x = self.nonlinear(x)
            x = self.conv_enc(x)

            mu, log_var = torch.chunk(x, 2, dim = 1)
            log_var = torch.clamp(log_var, -30.0, 20.0)
            std = torch.exp(0.5 * log_var)
            var = torch.exp(log_var)
            x1 = mu + std * torch.randn(mu.shape).to(device = mu.device)
        else:
            t = None
            x1 = latent

        x = self.mid_block3(x1)
        x = self.mid_attn2(x)
        x = self.mid_block4(x)
        
        for block1, block2, attn, conv1, upsample in self.ups:
            x = block1(x)
            x = block2(x)
            x = attn(x)
            #x = upsample(x)
            if isinstance(conv1, nn.Conv2d):
                em = conv1(x)
                lists.append(em)
            x = upsample(x)
            
        x2 = self.final_res_block(x)
        x2 = self.final_conv(x2)
        lists.append(x2)

        if latent is None:
            return lists, mu, var, x1
        else:
            return lists


class Unet_ab(nn.Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        dim_mults=(1, 1, 2, 2, 4, 4),
        channels = 3,
        resnet_block_groups = 8,
        input_size = 32,
        out_channels = 3,
        downsample_factor=3,
        upsample_factor=2
    ):
        super().__init__()

        # determine dimensions
        self.input_size = input_size
        self.channels = channels
        self.downsample_factor = downsample_factor
        self.upsample_factor = upsample_factor
        input_channels = channels

        init_dim = default(init_dim, dim)
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # layers
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # Downsampling
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_downsample = ind < self.downsample_factor
            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in),
                block_klass(dim_in, dim_in),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if is_downsample else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_dim = mid_dim
        self.mid_block1 = block_klass(mid_dim, mid_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim)

        self.mid_block3 = block_klass(mid_dim, mid_dim)
        self.mid_attn2 = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block4 = block_klass(mid_dim, mid_dim)

        self.normalize = nn.GroupNorm(16, mid_dim)
        self.nonlinear = nn.SiLU()
        self.conv_enc = nn.Conv2d(mid_dim, mid_dim * 2, 3, padding = 1)


        # Upsampling
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            idx_up = len(in_out) - 1 - self.upsample_factor
            is_upsample = ind >= idx_up and ind <= (idx_up + self.upsample_factor - 1)

            #idx_pe = len(in_out) - 2
            #is_last_enc = ind == idx_pe or ind == (idx_pe - 1)
            #is_last_enc = 0

            self.ups.append(nn.ModuleList([
                block_klass(dim_out, dim_out),
                block_klass(dim_out, dim_out),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                #nn.Conv2d(dim_out, out_channels, 1) if is_last_enc else nn.Identity(),
                Upsample(dim_out, dim_in) if is_upsample else  nn.Conv2d(dim_out, dim_in, 3, padding = 1),
            ]))

        self.final_res_block = block_klass(dim, dim)
        self.final_conv = nn.Conv2d(dim, out_channels, 1)

    def forward(self, x = None, latent = None):
        #lists = []
        if latent is None:
            x = self.init_conv(x)
            r = x.clone()
            h = []
            for block1, block2, attn, downsample in self.downs:
                x = block1(x)
                x = block2(x)
                x = attn(x)
                x = downsample(x)
            
            x = self.mid_block1(x)
            x = self.mid_attn(x)
            x = self.mid_block2(x)
            x = self.normalize(x)
            x = self.nonlinear(x)
            x = self.conv_enc(x)

            mu, log_var = torch.chunk(x, 2, dim = 1)
            log_var = torch.clamp(log_var, -30.0, 20.0)
            std = torch.exp(0.5 * log_var)
            var = torch.exp(log_var)
            x1 = mu + std * torch.randn(mu.shape).to(device = mu.device)
        else:
            t = None
            x1 = latent

        x = self.mid_block3(x1)
        x = self.mid_attn2(x)
        x = self.mid_block4(x)
        
        for block1, block2, attn, upsample in self.ups:
            x = block1(x)
            x = block2(x)
            x = attn(x)
            x = upsample(x)
            
        x2 = self.final_res_block(x)
        x2 = self.final_conv(x2)
        
        if latent is None:
            return x2, mu, var, x1
        else:
            return x2



def conv3d(in_channels, out_channels, kernel_size, bias, padding=1):
    return nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)


def create_conv(in_channels, out_channels, kernel_size, order, num_groups, padding=1):
    """
    Create a list of modules with together constitute a single conv layer with non-linearity
    and optional batchnorm/groupnorm.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        order (string): order of things, e.g.
            'cr' -> conv + ReLU
            'gcr' -> groupnorm + conv + ReLU
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
            'bcr' -> batchnorm + conv + ReLU
        num_groups (int): number of groups for the GroupNorm
        padding (int): add zero-padding to the input
    Return:
        list of tuple (name, module)
    """
    assert 'c' in order, "Conv layer MUST be present"
    assert order[0] not in 'rle', 'Non-linearity cannot be the first operation in the layer'

    modules = []
    for i, char in enumerate(order):
        if char == 'r':
            modules.append(('ReLU', nn.ReLU(inplace=True)))
        elif char == 'l':
            modules.append(('LeakyReLU', nn.LeakyReLU(negative_slope=0.1, inplace=True)))
        elif char == 'e':
            modules.append(('ELU', nn.ELU(inplace=True)))
        elif char == 'c':
            # add learnable bias only in the absence of batchnorm/groupnorm
            bias = not ('g' in order or 'b' in order)
            modules.append(('conv', conv3d(in_channels, out_channels, kernel_size, bias, padding=padding)))
        elif char == 'g':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                num_channels = in_channels
            else:
                num_channels = out_channels

            # use only one group if the given number of groups is greater than the number of channels
            if num_channels < num_groups:
                num_groups = 1

            assert num_channels % num_groups == 0, f'Expected number of channels in input to be divisible by num_groups. num_channels={num_channels}, num_groups={num_groups}'
            modules.append(('groupnorm', nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)))
        elif char == 'b':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                modules.append(('batchnorm', nn.BatchNorm3d(in_channels)))
            else:
                modules.append(('batchnorm', nn.BatchNorm3d(out_channels)))
        else:
            raise ValueError(f"Unsupported layer type '{char}'. MUST be one of ['b', 'g', 'r', 'l', 'e', 'c']")

    return modules

class Upsampling(nn.Module):
    """
    Upsamples a given multi-channel 3D data using either interpolation or learned transposed convolution.
    Args:
        transposed_conv (bool): if True uses ConvTranspose3d for upsampling, otherwise uses interpolation
        concat_joining (bool): if True uses concatenation joining between encoder and decoder features, otherwise
            uses summation joining (see Residual U-Net)
        in_channels (int): number of input channels for transposed conv
        out_channels (int): number of output channels for transpose conv
        kernel_size (int or tuple): size of the convolving kernel
        scale_factor (int or tuple): stride of the convolution
        mode (str): algorithm used for upsampling:
            'nearest' | 'linear' | 'bilinear' | 'trilinear' | 'area'. Default: 'nearest'
    """

    def __init__(self, transposed_conv, in_channels=None, out_channels=None, kernel_size=3,
                 scale_factor=(2, 2, 2), mode='nearest'):
        super(Upsampling, self).__init__()

        if transposed_conv:
            # make sure that the output size reverses the MaxPool3d from the corresponding encoder
            # (D_out = (D_in − 1) ×  stride[0] − 2 ×  padding[0] +  kernel_size[0] +  output_padding[0])
            self.upsample = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=scale_factor,
                                               padding=1)
        else:
            self.upsample = partial(self._interpolate, mode=mode)

    def forward(self, x, scale_factor=(2,1,1)):
        #output_size = encoder_features.size()[2:]
        return self.upsample(x, scale_factor = scale_factor)

    @staticmethod
    def _interpolate(x, mode, scale_factor):
        return F.interpolate(x, mode=mode, scale_factor=scale_factor)

class SingleConv(nn.Sequential):
    """
    Basic convolutional module consisting of a Conv3d, non-linearity and optional batchnorm/groupnorm. The order
    of operations can be specified via the `order` parameter
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, order='crg', num_groups=8, padding=1):
        super(SingleConv, self).__init__()

        for name, module in create_conv(in_channels, out_channels, kernel_size, order, num_groups, padding=padding):
            self.add_module(name, module)

class DoubleConv(nn.Sequential):
    """
    A module consisting of two consecutive convolution layers (e.g. BatchNorm3d+ReLU+Conv3d).
    We use (Conv3d+ReLU+GroupNorm3d) by default.
    This can be changed however by providing the 'order' argument, e.g. in order
    to change to Conv3d+BatchNorm3d+ELU use order='cbe'.
    Use padded convolutions to make sure that the output (H_out, W_out) is the same
    as (H_in, W_in), so that you don't have to crop in the decoder path.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        encoder (bool): if True we're in the encoder path, otherwise we're in the decoder
        kernel_size (int): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, encoder, kernel_size=3, order='crg', num_groups=8):
        super(DoubleConv, self).__init__()
        if encoder:
            # we're in the encoder path
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels // 2
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            # we're in the decoder path, decrease the number of channels in the 1st convolution
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        # conv1
        self.add_module('SingleConv1',
                        SingleConv(conv1_in_channels, conv1_out_channels, kernel_size, order, num_groups))
        # conv2
        self.add_module('SingleConv2',
                        SingleConv(conv2_in_channels, conv2_out_channels, kernel_size, order, num_groups))


class Encoder(nn.Module):
    """
    A single module from the encoder path consisting of the optional max
    pooling layer (one may specify the MaxPool kernel_size to be different
    than the standard (2,2,2), e.g. if the volumetric data is anisotropic
    (make sure to use complementary scale_factor in the decoder path) followed by
    a DoubleConv module.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        conv_kernel_size (int): size of the convolving kernel
        apply_pooling (bool): if True use MaxPool3d before DoubleConv
        pool_kernel_size (tuple): the size of the window to take a max over
        pool_type (str): pooling layer: 'max' or 'avg'
        basic_module(nn.Module): either ResNetBlock or DoubleConv
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, conv_kernel_size=3, apply_pooling=True,
                 pool_kernel_size=(2, 2, 2), pool_type='max', basic_module=DoubleConv, conv_layer_order='crg',
                 num_groups=8):
        super(Encoder, self).__init__()
        assert pool_type in ['max', 'avg']
        if apply_pooling:
            if pool_type == 'max':
                self.pooling = nn.MaxPool3d(kernel_size=pool_kernel_size)
            else:
                self.pooling = nn.AvgPool3d(kernel_size=pool_kernel_size)
        else:
            self.pooling = None

        self.basic_module = basic_module(in_channels, out_channels,
                                         encoder=True,
                                         kernel_size=conv_kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups)

    def forward(self, x):
        if self.pooling is not None:
            x = self.pooling(x)
        x = self.basic_module(x)
        return x



class Unet3D(nn.Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        dim_mults=(1, 1, 2, 2, 4, 4),
        channels = 3,
        resnet_block_groups = 8,
        input_size = 32,
        out_channels = 3,
        use_hdpe = True,
        downsample_factor = 3,
        basic_module = DoubleConv,
        layer_order = 'gcr',
        num_groups = 16,
    ):
        super().__init__()

        # determine dimensions
        self.use_hdpe = use_hdpe
        self.input_size = input_size
        self.channels = channels
        input_channels = channels

        init_dim = default(init_dim, dim)
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # layers
        self.init_conv = Encoder(input_channels, init_dim, apply_pooling = False, basic_module = basic_module,
                                conv_layer_order = layer_order, num_groups = num_groups)
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            # Downsampling
            is_downsample = ind < downsample_factor
            is_upsample = ind == 1
            self.downs.append(nn.ModuleList([
                Encoder(dim_in, dim_out, apply_pooling = is_downsample, pool_kernel_size=(1, 2, 2), 
                        basic_module = basic_module, conv_layer_order = layer_order, num_groups=num_groups),
                Upsampling(False) if is_upsample else nn.Identity()
            ]))

        # To calculate the dimension of latent variable
        temp_dim = dims[-1]
        temp_spatial = int(input_size // (2**downsample_factor))
        self.mid_dim = temp_dim * temp_spatial
        
        #TODO: more complicated projection function
        self.plane_xy = nn.Conv2d(self.mid_dim, int(temp_dim * 2), 1)
        self.plane_xt = nn.Conv2d(self.mid_dim, int(temp_dim * 2), 1)
        self.plane_yt = nn.Conv2d(self.mid_dim, int(temp_dim * 2), 1)

        #self.plane_xy = block_klass(self.mid_dim, int(temp_dim * 2))
        #self.plane_xt = block_klass(self.mid_dim, int(temp_dim * 2))
        #self.plane_yt = block_klass(self.mid_dim, int(temp_dim * 2))

        self.temp_dim = temp_dim


        self.mid_block = nn.Sequential(
            block_klass(temp_dim*3, temp_dim*3),
            Residual(PreNorm(temp_dim*3, Attention(temp_dim*3))),
            block_klass(temp_dim*3, temp_dim*3)
        )

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            #Upsample
            is_upsample = ind < downsample_factor 
            if not self.use_hdpe:
                is_last_enc = False
            else:
                is_last_enc = ind == (num_resolutions - downsample_factor) or ind == ((num_resolutions+1) - downsample_factor)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out, dim_out),
                block_klass(dim_out, dim_out),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                nn.Conv2d(dim_out, out_channels, 1) if is_last_enc else nn.Identity(),
                Upsample(dim_out, dim_in) if is_upsample else  nn.Conv2d(dim_out, dim_in, 3, padding = 1),
            ]))

        self.final_res_block = block_klass(dim, dim)
        self.final_conv = nn.Conv2d(dim, out_channels, 1)

    def _decode(self, x):  
        lists=[] 
        for block1, block2, attn, conv1, upsample in self.ups:
            x = block1(x)
            x = block2(x)
            x = attn(x)
            #x = upsample(x)
            if isinstance(conv1, nn.Conv2d):
                em = conv1(x)
                lists.append(em)
            x = upsample(x)
            
        x = self.final_res_block(x)
        x = self.final_conv(x)
        lists.append(x)
        return lists

    def decode(self, x):
        triplane = {}
        xy_latent = x[:, :self.mid_dim,:,:]
        xt_latent = x[:, self.mid_dim:self.mid_dim*2,:,:]
        yt_latent = x[:, self.mid_dim*2:,:,:]
        triplane['xy'] = self._decode(xy_latent)
        triplane['xt'] = self._decode(xt_latent)
        triplane['yt'] = self._decode(yt_latent)
        return triplane

    def forward(self, x = None, latent = None):
        lists = []
        if latent is None:
            x = self.init_conv(x)
            for encoder, up in self.downs:
                x = encoder(x)
                if isinstance(up, nn.Identity):
                    x = up(x)
                else:
                    x = up(x, scale_factor=(2,1,1))
    
            b, c, t, h, w = x.shape
            xy = x.reshape(b, c*t, h, w).contiguous()
            xt = x.permute(0, 1, 4, 2, 3).reshape(b, c*w, t, h).contiguous()
            yt = x.permute(0, 1, 3, 2, 4).reshape(b, c*h, t, w).contiguous()

            xy_latent = self.plane_xy(xy)
            xt_latent = self.plane_xt(xt)
            yt_latent = self.plane_yt(yt)

            mu_xy, log_var_xy = torch.chunk(xy_latent, 2, dim = 1)
            mu_xt, log_var_xt = torch.chunk(xt_latent, 2, dim = 1)
            mu_yt, log_var_yt = torch.chunk(yt_latent, 2, dim = 1)

            mu = torch.cat((mu_xy, mu_xt, mu_yt), dim = 1)
            log_var = torch.cat((log_var_xy, log_var_xt, log_var_yt), dim = 1)
            #mu, log_var = torch.chunk(x, 2, dim = 1)
            log_var = torch.clamp(log_var, -30.0, 20.0)
            std = torch.exp(0.5 * log_var)
            var = torch.exp(log_var)
            z = mu + std * torch.randn(mu.shape).to(device = mu.device)
        else:
            z = latent
          
        x =self.mid_block(z)
        triplane = {}
        xy_latent = x[:, :self.temp_dim,:,:]
        xt_latent = x[:, self.temp_dim:self.temp_dim*2,:,:]
        yt_latent = x[:, self.temp_dim*2:,:,:]
        triplane['xy'] = self._decode(xy_latent)
        triplane['xt'] = self._decode(xt_latent)
        triplane['yt'] = self._decode(yt_latent)
        
        if latent is None:
            return triplane, mu, var, z
        else:
            return triplane