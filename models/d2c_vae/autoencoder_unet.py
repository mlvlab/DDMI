import math
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from typing import Optional, Any

from models.ldm.modules.attention import LinearAttention
from models.ldm.modules.attention_efficient import MemoryEfficientCrossAttention
#from models.ldm.modules.diffusionmodules.openaimodel import AttentionBlock1D
from models.ldm.modules.distributions import DiagonalGaussianDistribution

try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False
    print("No module 'xformers'. Proceeding without it.")

def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0,1,0,0))
    return emb


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x, scale_factor=2.):
        x = torch.nn.functional.interpolate(x, scale_factor=scale_factor, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class LinAttnBlock(LinearAttention):
    """to match AttnBlock usage"""
    def __init__(self, in_channels):
        super().__init__(dim=in_channels, heads=1, dim_head=in_channels)


class AttnBlock(nn.Module):
    def __init__(self, in_channels, num_heads=4):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_channel = in_channels // num_heads
        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,-1)
        q = q.reshape(b*self.num_heads,self.head_channel, -1)
        q = q.permute(0,2,1)   # b,hw,c

        k = k.reshape(b,c,-1) # b,c,hw
        k = k.reshape(b*self.num_heads, self.head_channel, -1)
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        v = v.reshape(b * self.num_heads, self.head_channel, -1)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_

class AttnBlock1d(nn.Module):
    def __init__(self, in_channels, num_heads=4):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_channel = in_channels // num_heads
        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv1d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h = q.shape
        q = q.reshape(b * self.num_heads, self.head_channel, -1)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b * self.num_heads, self.head_channel, -1) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b * self.num_heads, self.head_channel, -1)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h)

        h_ = self.proj_out(h_)

        return x+h_

class MemoryEfficientAttnBlock(nn.Module):
    """
        Uses xformers efficient implementation,
        see https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
        Note: this is a single-head self-attention operation
    """
    #
    def __init__(self, in_channels, num_heads=4):
        super().__init__()
        self.in_channels = in_channels
        self.heads = num_heads
        self.head_channel = in_channels // num_heads
        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)
        self.attention_op: Optional[Any] = None

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        B, _, H, W = q.shape
        q, k, v = map(lambda x: rearrange(x, 'b c h w -> b (h w) c'), (q, k, v))

        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(B, t.shape[1], self.heads, self.head_channel)
            .permute(0, 2, 1, 3)
            .reshape(B * self.heads, t.shape[1], self.head_channel)
            .contiguous(),
            (q, k, v),
        )
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

        out = (
            out.unsqueeze(0)
            .reshape(B, self.heads, out.shape[1], self.head_channel)
            .permute(0, 2, 1, 3)
            .reshape(B, out.shape[1], self.heads* self.head_channel)
        )
        out = rearrange(out, 'b (h w) c -> b c h w', b=B, h=H, w=W, c=self.heads * self.head_channel)
        out = self.proj_out(out)
        return x+out
    
class MemoryEfficientAttnBlock_expand(nn.Module):
    """
        Uses xformers efficient implementation,
        see https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
        Note: this is a single-head self-attention operation
    """
    #
    def __init__(self, in_channels, num_heads=4):
        super().__init__()
        self.in_channels = in_channels
        self.heads = num_heads
        self.head_channel = in_channels
        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels*num_heads,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels*num_heads,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels*num_heads,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels*num_heads,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)
        self.attention_op: Optional[Any] = None

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        B, _, H, W = q.shape
        q, k, v = map(lambda x: rearrange(x, 'b c h w -> b (h w) c'), (q, k, v))

        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(B, t.shape[1], self.heads, self.head_channel)
            .permute(0, 2, 1, 3)
            .reshape(B * self.heads, t.shape[1], self.head_channel)
            .contiguous(),
            (q, k, v),
        )
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

        out = (
            out.unsqueeze(0)
            .reshape(B, self.heads, out.shape[1], self.head_channel)
            .permute(0, 2, 1, 3)
            .reshape(B, out.shape[1], self.heads* self.head_channel)
        )
        out = rearrange(out, 'b (h w) c -> b c h w', b=B, h=H, w=W, c=self.heads * self.head_channel)
        out = self.proj_out(out)
        return x+out

class MemoryEfficientAttnBlock1D(nn.Module):
    """
        Uses xformers efficient implementation,
        Originally from https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
        but modified to 1D case
    """
    #
    def __init__(self, in_channels, num_heads=8):
        super().__init__()
        self.in_channels = in_channels
        self.heads = num_heads
        self.head_channel = in_channels // num_heads
        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv1d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)
        self.attention_op: Optional[Any] = None

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        B, _, _  = q.shape
        q, k, v = map(lambda x: rearrange(x, 'b c h -> b h c'), (q, k, v))

        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(B, t.shape[1], self.heads, self.head_channel)
            .permute(0, 2, 1, 3)
            .reshape(B * self.heads, t.shape[1], self.head_channel)
            .contiguous(),
            (q, k, v),
        )
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

        out = (
            out.unsqueeze(0)
            .reshape(B, self.heads, out.shape[1], self.head_channel)
            .permute(0, 2, 1, 3)
            .reshape(B, out.shape[1], self.heads* self.head_channel)
        )
        out = rearrange(out, 'b h c -> b c h')
        out = self.proj_out(out)
        return x+out

class MemoryEfficientAttnBlock1D_expand(nn.Module):
    """
        Uses xformers efficient implementation,
        Originally from https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
        but modified to 1D case
    """
    #
    def __init__(self, in_channels, num_heads=8):
        super().__init__()
        self.in_channels = in_channels
        self.heads = num_heads
        self.head_channel = in_channels
        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv1d(in_channels,
                                 in_channels*num_heads,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv1d(in_channels,
                                 in_channels*num_heads,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv1d(in_channels,
                                 in_channels*num_heads,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv1d(in_channels*num_heads,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)
        self.attention_op: Optional[Any] = None

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        B, _, _  = q.shape
        q, k, v = map(lambda x: rearrange(x, 'b c h -> b h c'), (q, k, v))

        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(B, t.shape[1], self.heads, self.head_channel)
            .permute(0, 2, 1, 3)
            .reshape(B * self.heads, t.shape[1], self.head_channel)
            .contiguous(),
            (q, k, v),
        )
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

        out = (
            out.unsqueeze(0)
            .reshape(B, self.heads, out.shape[1], self.head_channel)
            .permute(0, 2, 1, 3)
            .reshape(B, out.shape[1], self.heads* self.head_channel)
        )
        out = rearrange(out, 'b h c -> b c h')
        out = self.proj_out(out)
        return x+out

class MemoryEfficientCrossAttentionWrapper(MemoryEfficientCrossAttention):
    def forward(self, x, context=None, mask=None):
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        out = super().forward(x, context=context, mask=mask)
        out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w, c=c)
        return x + out

'''
def make_attn(in_channels, attn_type="vanilla"):
    assert attn_type in ["vanilla", "linear", "none"], f'attn_type {attn_type} unknown'
    print(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == "vanilla":
        return AttnBlock(in_channels)
    elif attn_type == "none":
        return nn.Identity(in_channels)
    else:
        return LinAttnBlock(in_channels)
'''

def make_attn(in_channels, attn_type="vanilla", attn_kwargs=None):
    assert attn_type in ["vanilla", "vanilla-multihead", "vanilla-1d", "vanilla-1d-multihead", "memory-efficient-cross-attn", "linear", "none"], f'attn_type {attn_type} unknown'
    #if XFORMERS_IS_AVAILBLE and attn_type == "vanilla":
    #    attn_type = "vanilla-xformers"
    print(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == "vanilla":
        assert attn_kwargs is None
        if XFORMERS_IS_AVAILBLE:
            print(f"building MemoryEfficientAttnBlock with {in_channels} in_channels...")
            return MemoryEfficientAttnBlock(in_channels, num_heads=1)
        else:
            return AttnBlock(in_channels, num_heads=1)
    elif attn_type =='vanilla-multihead':
        if XFORMERS_IS_AVAILBLE:
            print(f"building MemoryEfficientAttnBlock with {in_channels} in_channels...")
            return MemoryEfficientAttnBlock(in_channels, num_heads=16)
        else:
            return AttnBlock(in_channels, num_heads=16)
    elif attn_type =='vanilla-multihead-expand':
        if XFORMERS_IS_AVAILBLE:
            print(f"building MemoryEfficientAttnBlock with {in_channels} in_channels...")
            return MemoryEfficientAttnBlock_expand(in_channels, num_heads=16)
        else:
            raise ValueError('Invalid without xformer packages')
    elif attn_type == "vanilla-1d":
        if XFORMERS_IS_AVAILBLE:
            print(f"building MemoryEfficientAttnBlock with {in_channels} in_channels...")
            return MemoryEfficientAttnBlock1D(in_channels, num_heads=1)
        else:
            return AttnBlock1d(in_channels, num_heads=1)
    elif attn_type == "vanilla-1d-multihead":
        if XFORMERS_IS_AVAILBLE:
            print(f"building MemoryEfficientAttnBlock with {in_channels} in_channels...")
            return MemoryEfficientAttnBlock1D(in_channels, num_heads=16)
        else:
            return AttnBlock1d(in_channels, num_heads=16)
    elif attn_type == "vanilla-1d-multihead-expand":
        if XFORMERS_IS_AVAILBLE:
            print(f"building MemoryEfficientAttnBlock with {in_channels} in_channels...")
            return MemoryEfficientAttnBlock1D_expand(in_channels, num_heads=4)
        else:
            raise ValueError('Invalid without xformer packages')
    elif type == "memory-efficient-cross-attn":
        attn_kwargs["query_dim"] = in_channels
        return MemoryEfficientCrossAttentionWrapper(**attn_kwargs)
    elif attn_type == "none":
        return nn.Identity(in_channels)
    else:
        raise NotImplementedError()


class Encoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, use_linear_attn=False, attn_type="vanilla",
                 **ignore_kwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, tanh_out=False, use_linear_attn=False,
                 attn_type="vanilla", hdbf_resolutions, temb_ch=0, use_timestep=False, **ignorekwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = temb_ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out
        self.use_timestep = use_timestep
        if self.use_timestep:
            self.temb = nn.Module()
            self.temb.dense = nn.ModuleList([
                torch.nn.Linear(self.ch,
                                self.temb_ch),
                torch.nn.Linear(self.temb_ch,
                                self.temb_ch),
            ])

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            hdbf = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))

            if curr_res in hdbf_resolutions:
                hdbf.append(nn.Conv2d(block_in, out_ch, 1))

            up = nn.Module()
            up.block = block
            up.attn = attn
            up.hdbf = hdbf
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z, t=None, plane=None):
        #assert z.shape[1:] == self.z_shape[1:]
        hdbf = []
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if len(self.up[i_level].hdbf) > 0:
                inter_bf = self.up[i_level].hdbf[0](h)
                hdbf.append(inter_bf)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        hdbf.append(h)
        #return h
        return hdbf


# ====================================================================================================

class Autoencoder(nn.Module):
    def __init__(self,
                 ddconfig,
                 embed_dim,
                 image_key="image",
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x


class Autoencoder3D(nn.Module):
    def __init__(self,
                 ddconfig,
                 embed_dim,
                 image_key="image",
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder_triplane(**ddconfig)
        self.decoder = Decoder_triplane(**ddconfig)
        assert ddconfig["double_z"]
        self.quant_conv_xy = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.quant_conv_yz = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.quant_conv_xz = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv_xy = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.post_quant_conv_yz = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.post_quant_conv_xz = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim

    def encode(self, x):
        xy, yz, xz = self.encoder(x)
        moments_xy = self.quant_conv_xy(xy)
        moments_yz = self.quant_conv_yz(yz)
        moments_xz = self.quant_conv_xz(xz)
        posterior_xy = DiagonalGaussianDistribution(moments_xy)
        posterior_yz = DiagonalGaussianDistribution(moments_yz)
        posterior_xz = DiagonalGaussianDistribution(moments_xz)

        return posterior_xy, posterior_yz, posterior_xz

    def decode(self, x):
        xy, yz, xz = x[0], x[1], x[2]
        xy = self.post_quant_conv_xy(xy)
        yz = self.post_quant_conv_yz(yz)
        xz = self.post_quant_conv_xz(xz)
        xy, yz, xz = self.decoder((xy, yz, xz))
        return xy, yz, xz

    def forward(self, input, sample_posterior=True):
        xy_posterior, yz_posterior, xz_posterior = self.encode(input)
        if sample_posterior:
            xy = xy_posterior.sample()
            yz = yz_posterior.sample()
            xz = xz_posterior.sample()
        else:
            xy = xy_posterior.mode()
            yz = yz_posterior.mode()
            xz = xz_posterior.mode()
        dec = self.decode(xy, yz, xz)
        return dec, xy_posterior, yz_posterior, xz_posterior

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

# ====================================================================================================

'''
Modules for tri-plane decoder
'''
class VideoDecoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, tanh_out=False, use_linear_attn=False,
                 attn_type="vanilla", hdbf_resolutions, inter_attn_resolutions, temb_ch=0, use_timestep=False, **ignorekwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = temb_ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out
        self.use_timestep = use_timestep
        if self.use_timestep:
            self.temb = nn.Module()
            self.temb.dense = nn.ModuleList([
                torch.nn.Linear(self.ch,
                                self.temb_ch),
                torch.nn.Linear(self.temb_ch,
                                self.temb_ch),
            ])

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        
        self.mid_attn = AttentionBlock1D(
                    block_in,
                    use_checkpoint=False,
                    num_heads=8,
                    num_head_channels=block_in // 8,
                    use_new_attention_order=False
                    )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            hdbf = nn.ModuleList()
            inter_attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
                
                if curr_res in inter_attn_resolutions:
                    inter_attn.append(AttentionBlock1D(
                        block_in,
                        use_checkpoint=False,
                        num_heads=8,
                        num_head_channels=block_in // 8,
                        use_new_attention_order=False,
                    ))
            if curr_res in hdbf_resolutions:
                hdbf.append(nn.Conv2d(block_in, out_ch, 1))

            up = nn.Module()
            up.block = block
            up.attn = attn
            up.hdbf = hdbf
            up.inter_attn = inter_attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z):
        #assert z.shape[1:] == self.z_shape[1:]
        hdbf_xy = []
        hdbf_yt = []
        hdbf_xt = []
        
        # timestep embedding
        temb = None

        h_xy = z[0]
        h_yt = z[1]
        h_xt = z[2]
        # z to block_in
        h_xy = self.conv_in(h_xy)
        h_yt = self.conv_in(h_yt)
        h_xt = self.conv_in(h_xt)

        # middle
        h_xy = self.mid.block_1(h_xy, temb)
        h_xy = self.mid.attn_1(h_xy)
        h_xy = self.mid.block_2(h_xy, temb)
        
        h_yt = self.mid.block_1(h_yt, temb)
        h_yt = self.mid.attn_1(h_yt)
        h_yt = self.mid.block_2(h_yt, temb)
        
        h_xt = self.mid.block_1(h_xt, temb)
        h_xt = self.mid.attn_1(h_xt)
        h_xt = self.mid.block_2(h_xt, temb)

        res = h_xy.size(-2)
        t   = h_xt.size(-2)

        h_xy = h_xy.view(h_xy.size(0), h_xy.size(1), -1)
        h_yt = h_yt.view(h_yt.size(0), h_yt.size(1), -1)
        h_xt = h_xt.view(h_xt.size(0), h_xt.size(1), -1)

        h = torch.cat([h_xy, h_xt, h_yt], dim=-1)
        h = self.mid_attn(h)

        h_xy = h[:, :, 0:res*res].view(h.size(0), h.size(1), res, res)
        h_xt = h[:, :, res*res:res*(res+t)].view(h.size(0), h.size(1), t, res)
        h_yt = h[:, :, res*(res+t):res*(res+t+t)].view(h.size(0), h.size(1), t, res)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h_xy = self.up[i_level].block[i_block](h_xy, temb)
                h_yt = self.up[i_level].block[i_block](h_yt, temb)
                h_xt = self.up[i_level].block[i_block](h_xt, temb)

                if len(self.up[i_level].attn) > 0:
                    h_xy = self.up[i_level].attn[i_block](h_xy)
                    h_yt = self.up[i_level].attn[i_block](h_yt)
                    h_xt = self.up[i_level].attn[i_block](h_xt)

                if len(self.up[i_level].inter_attn) > 0:
                    res = h_xy.size(-2)
                    t   = h_xt.size(-2)

                    h_xy = h_xy.view(h_xy.size(0), h_xy.size(1), -1)
                    h_yt = h_yt.view(h_yt.size(0), h_yt.size(1), -1)
                    h_xt = h_xt.view(h_xt.size(0), h_xt.size(1), -1)

                    h = torch.cat([h_xy, h_xt, h_yt], dim=-1)
                    h = self.up[i_level].inter_attn[i_block](h)

                    h_xy = h[:, :, 0:res*res].view(h.size(0), h.size(1), res, res)
                    h_xt = h[:, :, res*res:res*(res+t)].view(h.size(0), h.size(1), t, res)
                    h_yt = h[:, :, res*(res+t):res*(res+t+t)].view(h.size(0), h.size(1), t, res)

            if len(self.up[i_level].hdbf) > 0:
                inter_bf_xy = self.up[i_level].hdbf[0](h_xy)
                inter_bf_yt = self.up[i_level].hdbf[0](h_yt)
                inter_bf_xt = self.up[i_level].hdbf[0](h_xt)
                hdbf_xy.append(inter_bf_xy)
                hdbf_yt.append(inter_bf_yt)
                hdbf_xt.append(inter_bf_xt)
            
            if i_level != 0:
                h_xy = self.up[i_level].upsample(h_xy, scale_factor=2.)
                h_yt = self.up[i_level].upsample(h_yt, scale_factor=(1., 2.))
                h_xt = self.up[i_level].upsample(h_xt, scale_factor=(1., 2.))

        # end
        if self.give_pre_end:
            return h

        h_xy = self.norm_out(h_xy)
        h_yt = self.norm_out(h_yt)
        h_xt = self.norm_out(h_xt)

        h_xy = nonlinearity(h_xy)
        h_yt = nonlinearity(h_yt)
        h_xt = nonlinearity(h_xt)

        h_xy = self.conv_out(h_xy)
        h_yt = self.conv_out(h_yt)
        h_xt = self.conv_out(h_xt)

        hdbf_xy.append(h_xy)
        hdbf_yt.append(h_yt)
        hdbf_xt.append(h_xt)

        return hdbf_xy, hdbf_yt, hdbf_xt


class VideoDecoder_light(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, tanh_out=False, use_linear_attn=False,
                 attn_type="vanilla", hdbf_resolutions, inter_attn_resolutions, temb_ch=0, use_timestep=False, **ignorekwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = temb_ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out
        self.use_timestep = use_timestep
        if self.use_timestep:
            self.temb = nn.Module()
            self.temb.dense = nn.ModuleList([
                torch.nn.Linear(self.ch,
                                self.temb_ch),
                torch.nn.Linear(self.temb_ch,
                                self.temb_ch),
            ])

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        
        # From PVDM
        #self.mid_attn = AttentionBlock1D(
        #            block_in,
        #            use_checkpoint=False,
        #            num_heads=8,
        #            num_head_channels=block_in // 8,
        #            use_new_attention_order=False
        #            )

        # modify
        self.mid_attn = make_attn(block_in, attn_type='vanilla-1d-multihead')

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            hdbf = nn.ModuleList()
            inter_attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
                
            if curr_res in inter_attn_resolutions:
                inter_attn.append(make_attn(block_in, attn_type='vanilla-1d-multihead'))

            if curr_res in hdbf_resolutions:
                hdbf.append(nn.Conv2d(block_in, out_ch, 1))

            up = nn.Module()
            up.block = block
            up.attn = attn
            up.hdbf = hdbf
            up.inter_attn = inter_attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z):
        #assert z.shape[1:] == self.z_shape[1:]
        hdbf_xy = []
        hdbf_yt = []
        hdbf_xt = []
        
        # timestep embedding
        temb = None

        h_xy = z[0]
        h_yt = z[1]
        h_xt = z[2]
        # z to block_in
        h_xy = self.conv_in(h_xy)
        h_yt = self.conv_in(h_yt)
        h_xt = self.conv_in(h_xt)

        # middle
        h_xy = self.mid.block_1(h_xy, temb)
        h_xy = self.mid.attn_1(h_xy)
        h_xy = self.mid.block_2(h_xy, temb)
        
        h_yt = self.mid.block_1(h_yt, temb)
        h_yt = self.mid.attn_1(h_yt)
        h_yt = self.mid.block_2(h_yt, temb)
        
        h_xt = self.mid.block_1(h_xt, temb)
        h_xt = self.mid.attn_1(h_xt)
        h_xt = self.mid.block_2(h_xt, temb)

        res = h_xy.size(-2)
        t   = h_xt.size(-2)

        h_xy = h_xy.view(h_xy.size(0), h_xy.size(1), -1)
        h_yt = h_yt.view(h_yt.size(0), h_yt.size(1), -1)
        h_xt = h_xt.view(h_xt.size(0), h_xt.size(1), -1)

        h = torch.cat([h_xy, h_xt, h_yt], dim=-1)
        h = self.mid_attn(h)

        h_xy = h[:, :, 0:res*res].view(h.size(0), h.size(1), res, res)
        h_xt = h[:, :, res*res:res*(res+t)].view(h.size(0), h.size(1), t, res)
        h_yt = h[:, :, res*(res+t):res*(res+t+t)].view(h.size(0), h.size(1), t, res)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h_xy = self.up[i_level].block[i_block](h_xy, temb)
                h_yt = self.up[i_level].block[i_block](h_yt, temb)
                h_xt = self.up[i_level].block[i_block](h_xt, temb)

                if len(self.up[i_level].attn) > 0:
                    h_xy = self.up[i_level].attn[i_block](h_xy)
                    h_yt = self.up[i_level].attn[i_block](h_yt)
                    h_xt = self.up[i_level].attn[i_block](h_xt)

            if len(self.up[i_level].inter_attn) > 0:
                res = h_xy.size(-2)
                t   = h_xt.size(-2)

                h_xy = h_xy.view(h_xy.size(0), h_xy.size(1), -1)
                h_yt = h_yt.view(h_yt.size(0), h_yt.size(1), -1)
                h_xt = h_xt.view(h_xt.size(0), h_xt.size(1), -1)

                h = torch.cat([h_xy, h_xt, h_yt], dim=-1)
                h = self.up[i_level].inter_attn[0](h)

                h_xy = h[:, :, 0:res*res].view(h.size(0), h.size(1), res, res)
                h_xt = h[:, :, res*res:res*(res+t)].view(h.size(0), h.size(1), t, res)
                h_yt = h[:, :, res*(res+t):res*(res+t+t)].view(h.size(0), h.size(1), t, res)

            if len(self.up[i_level].hdbf) > 0:
                inter_bf_xy = self.up[i_level].hdbf[0](h_xy)
                inter_bf_yt = self.up[i_level].hdbf[0](h_yt)
                inter_bf_xt = self.up[i_level].hdbf[0](h_xt)
                hdbf_xy.append(inter_bf_xy)
                hdbf_yt.append(inter_bf_yt)
                hdbf_xt.append(inter_bf_xt)
            
            if i_level != 0:
                h_xy = self.up[i_level].upsample(h_xy, scale_factor=2.)
                h_yt = self.up[i_level].upsample(h_yt, scale_factor=(1., 2.)) # not upsample t
                h_xt = self.up[i_level].upsample(h_xt, scale_factor=(1., 2.)) # not upsample t

        # end
        if self.give_pre_end:
            return h

        h_xy = self.norm_out(h_xy)
        h_yt = self.norm_out(h_yt)
        h_xt = self.norm_out(h_xt)

        h_xy = nonlinearity(h_xy)
        h_yt = nonlinearity(h_yt)
        h_xt = nonlinearity(h_xt)

        h_xy = self.conv_out(h_xy)
        h_yt = self.conv_out(h_yt)
        h_xt = self.conv_out(h_xt)

        hdbf_xy.append(h_xy)
        hdbf_yt.append(h_yt)
        hdbf_xt.append(h_xt)

        return hdbf_xy, hdbf_yt, hdbf_xt
    

class Encoder_triplane(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z= True, give_pre_end=False, tanh_out=False, use_linear_attn=False,
                 attn_type="vanilla", inter_attn_resolutions, temb_ch=0, use_timestep=False, **ignorekwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = temb_ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out
        self.use_timestep = use_timestep
        if self.use_timestep:
            self.temb = nn.Module()
            self.temb.dense = nn.ModuleList([
                torch.nn.Linear(self.ch,
                                self.temb_ch),
                torch.nn.Linear(self.temb_ch,
                                self.temb_ch),
            ])

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)

        # z to block_in
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # modify
        self.mid_attn = make_attn(block_in, attn_type='vanilla-1d')

        # upsampling
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            inter_attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
                
            if curr_res in inter_attn_resolutions:
                inter_attn.append(make_attn(block_in, attn_type='vanilla-1d'))

            down = nn.Module()
            down.block = block
            down.attn = attn
            down.inter_attn = inter_attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.down.insert(0, down) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z):
        # timestep embedding
        temb = None

        h_xy = z[0]
        h_yt = z[1]
        h_xt = z[2]
        # z to block_in
        h_xy = self.conv_in(h_xy)
        h_yt = self.conv_in(h_yt)
        h_xt = self.conv_in(h_xt)

        # middle
        h_xy = self.mid.block_1(h_xy, temb)
        h_xy = self.mid.attn_1(h_xy)
        h_xy = self.mid.block_2(h_xy, temb)
        
        h_yt = self.mid.block_1(h_yt, temb)
        h_yt = self.mid.attn_1(h_yt)
        h_yt = self.mid.block_2(h_yt, temb)
        
        h_xt = self.mid.block_1(h_xt, temb)
        h_xt = self.mid.attn_1(h_xt)
        h_xt = self.mid.block_2(h_xt, temb)

        res = h_xy.size(-2)
        t   = h_xt.size(-2)

        h_xy = h_xy.view(h_xy.size(0), h_xy.size(1), -1)
        h_yt = h_yt.view(h_yt.size(0), h_yt.size(1), -1)
        h_xt = h_xt.view(h_xt.size(0), h_xt.size(1), -1)

        h = torch.cat([h_xy, h_xt, h_yt], dim=-1)
        h = self.mid_attn(h)

        h_xy = h[:, :, 0:res*res].view(h.size(0), h.size(1), res, res)
        h_xt = h[:, :, res*res:res*(res+t)].view(h.size(0), h.size(1), t, res)
        h_yt = h[:, :, res*(res+t):res*(res+t+t)].view(h.size(0), h.size(1), t, res)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h_xy = self.down[i_level].block[i_block](h_xy, temb)
                h_yt = self.down[i_level].block[i_block](h_yt, temb)
                h_xt = self.down[i_level].block[i_block](h_xt, temb)

                if len(self.down[i_level].attn) > 0:
                    h_xy = self.down[i_level].attn[i_block](h_xy)
                    h_yt = self.down[i_level].attn[i_block](h_yt)
                    h_xt = self.down[i_level].attn[i_block](h_xt)

            if len(self.down[i_level].inter_attn) > 0:
                res = h_xy.size(-2)
                t   = h_xt.size(-2)

                h_xy = h_xy.view(h_xy.size(0), h_xy.size(1), -1)
                h_yt = h_yt.view(h_yt.size(0), h_yt.size(1), -1)
                h_xt = h_xt.view(h_xt.size(0), h_xt.size(1), -1)

                h = torch.cat([h_xy, h_xt, h_yt], dim=-1)
                h = self.down[i_level].inter_attn[0](h)

                h_xy = h[:, :, 0:res*res].view(h.size(0), h.size(1), res, res)
                h_xt = h[:, :, res*res:res*(res+t)].view(h.size(0), h.size(1), t, res)
                h_yt = h[:, :, res*(res+t):res*(res+t+t)].view(h.size(0), h.size(1), t, res)
            
            if i_level != 0:
                h_xy = self.down[i_level].downsample(h_xy)
                h_yt = self.down[i_level].downsample(h_yt)
                h_xt = self.down[i_level].downsample(h_xt)

        # end
        if self.give_pre_end:
            return h

        h_xy = self.norm_out(h_xy)
        h_yt = self.norm_out(h_yt)
        h_xt = self.norm_out(h_xt)

        h_xy = nonlinearity(h_xy)
        h_yt = nonlinearity(h_yt)
        h_xt = nonlinearity(h_xt)

        h_xy = self.conv_out(h_xy)
        h_yt = self.conv_out(h_yt)
        h_xt = self.conv_out(h_xt)
        return h_xy, h_yt, h_xt
    

class Decoder_triplane(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, tanh_out=False, use_linear_attn=False,
                 attn_type="vanilla", hdbf_resolutions, inter_attn_resolutions, temb_ch=0, use_timestep=False, **ignorekwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = temb_ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out
        self.use_timestep = use_timestep
        if self.use_timestep:
            self.temb = nn.Module()
            self.temb.dense = nn.ModuleList([
                torch.nn.Linear(self.ch,
                                self.temb_ch),
                torch.nn.Linear(self.temb_ch,
                                self.temb_ch),
            ])

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # modify
        self.mid_attn = make_attn(block_in, attn_type='vanilla-1d')

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            hdbf = nn.ModuleList()
            inter_attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
                
            if curr_res in inter_attn_resolutions:
                inter_attn.append(make_attn(block_in, attn_type='vanilla-1d'))

            if curr_res in hdbf_resolutions:
                hdbf.append(nn.Conv2d(block_in, out_ch, 1))

            up = nn.Module()
            up.block = block
            up.attn = attn
            up.hdbf = hdbf
            up.inter_attn = inter_attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z):
        #assert z.shape[1:] == self.z_shape[1:]
        hdbf_xy = []
        hdbf_yt = []
        hdbf_xt = []
        
        # timestep embedding
        temb = None

        h_xy = z[0]
        h_yt = z[1]
        h_xt = z[2]
        # z to block_in
        h_xy = self.conv_in(h_xy)
        h_yt = self.conv_in(h_yt)
        h_xt = self.conv_in(h_xt)

        # middle
        h_xy = self.mid.block_1(h_xy, temb)
        h_xy = self.mid.attn_1(h_xy)
        h_xy = self.mid.block_2(h_xy, temb)
        
        h_yt = self.mid.block_1(h_yt, temb)
        h_yt = self.mid.attn_1(h_yt)
        h_yt = self.mid.block_2(h_yt, temb)
        
        h_xt = self.mid.block_1(h_xt, temb)
        h_xt = self.mid.attn_1(h_xt)
        h_xt = self.mid.block_2(h_xt, temb)

        res = h_xy.size(-2)
        t   = h_xt.size(-2)

        h_xy = h_xy.view(h_xy.size(0), h_xy.size(1), -1)
        h_yt = h_yt.view(h_yt.size(0), h_yt.size(1), -1)
        h_xt = h_xt.view(h_xt.size(0), h_xt.size(1), -1)

        h = torch.cat([h_xy, h_xt, h_yt], dim=-1)
        h = self.mid_attn(h)

        h_xy = h[:, :, 0:res*res].view(h.size(0), h.size(1), res, res)
        h_xt = h[:, :, res*res:res*(res+t)].view(h.size(0), h.size(1), t, res)
        h_yt = h[:, :, res*(res+t):res*(res+t+t)].view(h.size(0), h.size(1), t, res)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h_xy = self.up[i_level].block[i_block](h_xy, temb)
                h_yt = self.up[i_level].block[i_block](h_yt, temb)
                h_xt = self.up[i_level].block[i_block](h_xt, temb)

                if len(self.up[i_level].attn) > 0:
                    h_xy = self.up[i_level].attn[i_block](h_xy)
                    h_yt = self.up[i_level].attn[i_block](h_yt)
                    h_xt = self.up[i_level].attn[i_block](h_xt)

            if len(self.up[i_level].inter_attn) > 0:
                res = h_xy.size(-2)
                t   = h_xt.size(-2)

                h_xy = h_xy.view(h_xy.size(0), h_xy.size(1), -1)
                h_yt = h_yt.view(h_yt.size(0), h_yt.size(1), -1)
                h_xt = h_xt.view(h_xt.size(0), h_xt.size(1), -1)

                h = torch.cat([h_xy, h_xt, h_yt], dim=-1)
                h = self.up[i_level].inter_attn[0](h)

                h_xy = h[:, :, 0:res*res].view(h.size(0), h.size(1), res, res)
                h_xt = h[:, :, res*res:res*(res+t)].view(h.size(0), h.size(1), t, res)
                h_yt = h[:, :, res*(res+t):res*(res+t+t)].view(h.size(0), h.size(1), t, res)

            if len(self.up[i_level].hdbf) > 0:
                inter_bf_xy = self.up[i_level].hdbf[0](h_xy)
                inter_bf_yt = self.up[i_level].hdbf[0](h_yt)
                inter_bf_xt = self.up[i_level].hdbf[0](h_xt)
                hdbf_xy.append(inter_bf_xy)
                hdbf_yt.append(inter_bf_yt)
                hdbf_xt.append(inter_bf_xt)
            
            if i_level != 0:
                h_xy = self.up[i_level].upsample(h_xy, scale_factor=2.)
                h_yt = self.up[i_level].upsample(h_yt, scale_factor=2.)
                h_xt = self.up[i_level].upsample(h_xt, scale_factor=2.)

        # end
        if self.give_pre_end:
            return h

        h_xy = self.norm_out(h_xy)
        h_yt = self.norm_out(h_yt)
        h_xt = self.norm_out(h_xt)

        h_xy = nonlinearity(h_xy)
        h_yt = nonlinearity(h_yt)
        h_xt = nonlinearity(h_xt)

        h_xy = self.conv_out(h_xy)
        h_yt = self.conv_out(h_yt)
        h_xt = self.conv_out(h_xt)

        hdbf_xy.append(h_xy)
        hdbf_yt.append(h_yt)
        hdbf_xt.append(h_xt)

        return hdbf_xy, hdbf_yt, hdbf_xt