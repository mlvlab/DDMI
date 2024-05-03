import torch
import numpy as np
import random

import torch.nn.functional as F
import torchvision.transforms.functional as trans_F

def exists(x):
    return x is not None

def cycle(dl):
    while True:
        for data in dl:
            yield data

def random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


# Coordinate functions =========================================================================================================
def convert_to_coord_format_2d(b, h, w, device = 'cpu', integer_values = False, hstart = -1, hend = 1, wstart = -1, wend = 1):
    if integer_values:
        x_channel = torch.arange(w, dtype = torch.float, device = device).view(1, 1, 1, -1).repeat(b, 1, w, 1)
        y_channel = torch.arange(h, dtype = torch.float, device = device).view(1, 1, -1, 1).repeat(b, 1, 1, h)
    else:
        x_channel = torch.linspace(wstart, wend, w, device = device).view(1, 1, 1, -1).repeat(b, 1, w, 1)
        y_channel = torch.linspace(hstart, hend, h, device = device).view(1, 1, -1, 1).repeat(b, 1, 1, h)

    return torch.cat((x_channel, y_channel), dim = 1)


def convert_to_coord_format_3d(b, h, w, t, device = 'cpu', 
                                hstart = -1, hend = 1, wstart = -1, wend = 1,
                                tstart = -1, tend = 1):
    triplane_coords = {}
    x_channel = torch.linspace(wstart, wend, w, device = device).view(1, 1, 1, w).repeat(b, 1, h, 1)
    y_channel = torch.linspace(hstart, hend, h, device = device).view(1, 1, h, 1).repeat(b, 1, 1, w)
    triplane_coords['xy'] = torch.cat((x_channel, y_channel), dim = 1)
    x_channel = torch.linspace(wstart, wend, w, device = device).view(1, 1, 1, w).repeat(b, 1, t, 1)
    t_channel = torch.linspace(tstart, tend, t, device = device).view(1, 1, t, 1).repeat(b, 1, 1, w)
    triplane_coords['xt'] = torch.cat((t_channel, x_channel), dim = 1)
    y_channel = torch.linspace(hstart, hend, h, device = device).view(1, 1, 1, h).repeat(b, 1, t, 1)
    t_channel = torch.linspace(tstart, tend, t, device = device).view(1, 1, t, 1).repeat(b, 1, 1, h)
    triplane_coords['yt'] = torch.cat((t_channel, y_channel), dim = 1)

    return triplane_coords

def coordinate2index(x, reso, coord_type='2d'):
    ''' Normalize coordinate to [0, 1] for unit cube experiments.
        Corresponds to our 3D model

    Args:
        x (tensor): coordinate
        reso (int): defined resolution
        coord_type (str): coordinate type
    '''
    x = (x * reso).long()
    if coord_type == '2d': # plane
        index = x[:, :, 0] + reso * x[:, :, 1]
    elif coord_type == '3d': # grid
        index = x[:, :, 0] + reso * (x[:, :, 1] + reso * x[:, :, 2])
    index = index[:, None, :]
    return index

def normalize_coordinate(p, padding=0.1, plane='xz'):
    ''' Normalize coordinate to [0, 1] for unit cube experiments

    Args:
        p (tensor): point
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        plane (str): plane feature type, ['xz', 'xy', 'yz']
    '''
    if plane == 'xz':
        xy = p[:, :, [0, 2]]
    elif plane =='xy':
        xy = p[:, :, [0, 1]]
    else:
        xy = p[:, :, [1, 2]]

    xy_new = xy / (1 + padding + 10e-6) # (-0.5, 0.5)
    xy_new = xy_new + 0.5 # range (0, 1)

    # f there are outliers out of the range
    if xy_new.max() >= 1:
        xy_new[xy_new >= 1] = 1 - 10e-6
    if xy_new.min() < 0:
        xy_new[xy_new < 0] = 0.0
    return xy_new

def normalize_3d_coordinate(p, padding=0.1):
    ''' Normalize coordinate to [0, 1] for unit cube experiments.
        Corresponds to our 3D model

    Args:
        p (tensor): point
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    '''
    
    p_nor = p / (1 + padding + 10e-4) # (-0.5, 0.5)
    p_nor = p_nor + 0.5 # range (0, 1)
    # f there are outliers out of the range
    if p_nor.max() >= 1:
        p_nor[p_nor >= 1] = 1 - 10e-4
    if p_nor.min() < 0:
        p_nor[p_nor < 0] = 0.0
    return p_nor


def sample_plane_feature(p, plane='xz'):
        xy = normalize_coordinate(p.clone(), plane=plane, padding=0.1) # normalize to the range of (0, 1)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0 # normalize to (-1, 1)
        return vgrid


def singleplane_positional_encoding(hdbf, coords):
    return F.grid_sample(hdbf, coords, padding_mode='border')


def triplane_positional_encoding(hdbf1, hdbf2, hdbf3, coords1, coords2, coords3, mode = 'add'):
    # for planes with identical dimensionality
    if mode == 'add':
        x = F.grid_sample(hdbf1, coords1, padding_mode='border', align_corners=True, mode='bilinear').squeeze(-1)
        x += F.grid_sample(hdbf2, coords2, padding_mode='border', align_corners=True, mode='bilinear').squeeze(-1)
        x += F.grid_sample(hdbf3, coords3, padding_mode='border', align_corners=True, mode='bilinear').squeeze(-1)
    
    # for planes with different dimensionality
    elif mode == 'concat':
        x1 = F.grid_sample(hdbf1, coords1, padding_mode='border', align_corners=True, mode='bilinear')
        x2 = F.grid_sample(hdbf2, coords2, padding_mode='border', align_corners=True, mode='bilinear')
        x3 = F.grid_sample(hdbf3, coords3, padding_mode='border', align_corners=True, mode='bilinear')

        b, c, h, w = x1.shape
        _, _, t, _ = x2.shape
        x1 = x1.unsqueeze(2).repeat(1, 1, t, 1, 1)
        x2 = x2.unsqueeze(-1).repeat(1, 1, 1, 1, w)
        x3 = x3.unsqueeze(3).repeat(1, 1, 1, h, 1)
        x = torch.cat((x1, x2, x3), dim = 1).reshape(b, c*3, -1)
        x = x.permute(0, 2, 1).reshape(-1, c*3).contiguous()
    else:
        raise NotImplementedError
    return x


def multiscale_image_transform(x, size, multiscale, device):
    h_coordinate = convert_to_coord_format_2d(1, 512, 512, device = device, hstart=-511/512, hend=511/512, wstart=-511/512, wend=511/512)
    m_coordinate = convert_to_coord_format_2d(1, 384, 384, device = device, hstart=-383/384, hend=383/384, wstart=-383/384, wend=383/384)
    l_coordinate = convert_to_coord_format_2d(1, 256, 256, device = device, hstart=-255/256, hend = 255/256, wstart=-255/256, wend = 255/256)

    if multiscale:
        # 512x512
        if x.shape[2] > 512:
            y1 = trans_F.resize(x, 512, antialias = True)
            y1 = y1.clamp(-1., 1.)
        else:
            y1 = x.clone()    
        i = torch.randint(0, 511 - size + 1, size = (1,)).item()
        j = torch.randint(0, 511 - size + 1, size = (1,)).item()
        y1 = trans_F.crop(y1, i, j, size, size)

        # 384x384
        y2 = trans_F.resize(x, 384, antialias = True)
        y2 = y2.clamp(-1, 1.)
        i2 = torch.randint(0, 383 - size + 1, size = (1,)).item()
        j2 = torch.randint(0, 383 - size + 1, size = (1,)).item()
        y2 = trans_F.crop(y2, i2, j2, size, size)

        # 256x256
        y = trans_F.resize(x, 256, antialias = True)
        y = y.clamp(-1., 1.)

        p = random.random()

        if p <= 0.3:
            relative_scale = 1.
            target = y
            _, c, h, w = target.shape
            coordinate = convert_to_coord_format_2d(1, h, w, device = device, hstart=-255/256, hend = 255/256, wstart=-255/256, wend = 255/256)
        elif 0.3 < p <= 0.6:
            relative_scale = 1/1.5
            target = y2
            coordinate = trans_F.crop(m_coordinate, i2, j2, size, size)
        else:
            relative_scale = 1/2
            target = y1
            coordinate = trans_F.crop(h_coordinate, i, j, size, size)

    else:
        y = trans_F.resize(x, 256, antialias = True)
        y = y.clamp(-1., 1.)
        target = y
        coordinate = l_coordinate
        relative_scale = 1.
        
    return target, coordinate, relative_scale, y


def get_scale_injection(current_res, anchor_res=256):
    scale = anchor_res / current_res
    return scale

# ====================================================================================================================

def symmetrize_image_data(images):
    return 2.0 * images - 1.0


def unsymmetrize_image_data(images):
    return (images + 1.) / 2.


def linear_kl_coeff(step, total_step, constant_step, min_coeff, max_coeff):
    return max(min(min_coeff + (max_coeff - min_coeff) * (step - constant_step) / total_step, max_coeff), min_coeff)
