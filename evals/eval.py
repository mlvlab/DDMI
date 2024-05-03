import time
import sys; sys.path.extend(['.', 'src'])
import numpy as np
import torch
from torchvision.utils import save_image, make_grid
from einops import rearrange
import torchvision.utils as vtils
from PIL import Image

from evals.fvd.fvd import get_fvd_logits, frechet_distance
from evals.fvd.download import load_i3d_pretrained
from evals.fid.inception import InceptionV3
from evals.fid.fid_score import calculate_frechet_distance
from utils.general_utils import symmetrize_image_data, unsymmetrize_image_data, get_scale_injection
import torchvision.transforms.functional as trans_F
import os

import torchvision
import PIL

def save_image_grid(img, fname, drange, grid_size, normalize=True):
    if normalize:
        lo, hi = drange
        img = np.asarray(img, dtype=np.float32)
        img = (img - lo) * (255 / (hi - lo))
        img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, T, H, W = img.shape
    img = img.reshape(gh, gw, C, T, H, W)
    img = img.transpose(3, 0, 4, 1, 5, 2)
    img = img.reshape(T, gh * H, gw * W, C)

    print (f'Saving Video with {T} frames, img shape {H}, {W}')

    assert C in [3]

    if C == 3:
        torchvision.io.write_video(f'{fname[:-3]}mp4', torch.from_numpy(img), fps=16)
        imgs = [PIL.Image.fromarray(img[i], 'RGB') for i in range(len(img))]
        imgs[0].save(fname, quality=95, save_all=True, append_images=imgs[1:], duration=100, loop=0)

    return img

def test_psnr(rank, model, loader, it, logger=None):
    device = torch.device('cuda', rank)

    losses = dict()
    losses['psnr'] = AverageMeter()
    check = time.time()

    model.eval()
    with torch.no_grad():
        for n, (x, _) in enumerate(loader):
            if n > 100:
                break
            batch_size = x.size(0)
            clip_length = x.size(1)
            x = x.to(device) / 127.5 - 1
            recon, _ = model(rearrange(x, 'b t c h w -> b c t h w'))

            x = x.view(batch_size, -1)
            recon = recon.view(batch_size, -1)

            mse = ((x * 0.5 - recon * 0.5) ** 2).mean(dim=-1)
            psnr = (-10 * torch.log10(mse)).mean()

            losses['psnr'].update(psnr.item(), batch_size)


    model.train()
    return losses['psnr'].average


### Image evaluation metric

def test_rfid(vaemodel, mlp, coords, loader, path, device, save):
    check = time.time()

    real_embeddings = []
    fake_embeddings = []
    fakes = []
    reals = []

    vaemodel.eval()
    mlp.eval()
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx]).to(device)
    model.eval()

    with torch.no_grad():
        for n, (real, idx) in enumerate(loader):
            if n > 512:
                break
            batch_size = real.size(0)
            real = real.to(device)
            real = symmetrize_image_data(real)
            if isinstance(vaemodel, torch.nn.parallel.DistributedDataParallel):
                xy = vaemodel.module.encode(real)
                xy = vaemodel.module.decode(xy.sample())
            else:
                xy = vaemodel.encode(real)
                xy = vaemodel.decode(xy.sample())
            fake = mlp(coords, hdbf = xy, si = 1)
            fake = fake.clamp(-1,1)
            
            if save:
                vtils.save_image(fake, os.path.join(path, '2Test{}.png'.format(n)), normalize=True, scale_each=True)
            
            pred_real = model(real)[0]
            pred_real = pred_real.squeeze(3).squeeze(2).cpu().numpy()
            real_embeddings.append(pred_real)
            pred_fake = model(fake)[0]
            pred_fake = pred_fake.squeeze(3).squeeze(2).cpu().numpy()
            fake_embeddings.append(pred_fake)

    vaemodel.train()
    mlp.train()

    real_embeddings = np.concatenate(real_embeddings, axis=0)
    real_mu = np.mean(real_embeddings, axis=0)
    real_sigma = np.cov(real_embeddings, rowvar=False)

    fake_embeddings = np.concatenate(fake_embeddings, axis=0)
    fake_mu = np.mean(fake_embeddings, axis=0)
    fake_sigma = np.cov(fake_embeddings, rowvar=False)

    print('Total number of samples:', real_embeddings.shape[0])
    
    fid_value = calculate_frechet_distance(real_mu, real_sigma, fake_mu, fake_sigma)
    return fid_value

def test_fid_ddpm(ema, vaemodel, mlp, coords, loader, accelerator, path=None, save=False):
    real_embeddings = []
    fake_embeddings = []
    fakes = []
    reals = []

    device = accelerator.device    
    ema.ema_model.eval()
    vaemodel.eval()
    mlp.eval()
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx]).to(device)
    model.eval()
    with torch.inference_mode():
        for n, (real, idx) in enumerate(loader):
            if n > 21:
                break
            batch_size = real.size(0)
            shape = (batch_size, 64, 64, 64)
            real = real.to(device)
            real = symmetrize_image_data(real)
            with accelerator.autocast():
                z_test = ema.ema_model.sample(shape=shape)
                if isinstance(vaemodel, torch.nn.parallel.DistributedDataParallel):
                    pe_test = vaemodel.module.decode(z_test)
                else:
                    pe_test = vaemodel.decode(z_test)
                fake = mlp(coords, hdbf=pe_test, si=1)
            #fake = (fake.clamp(-1,1) + 1) * 127.5
            fake = fake.clamp(-1., 1.)
            if save:
                assert path is not None
                vtils.save_image(fake, os.path.join((path), 'Test-{}.png'.format(idx)), normalize = True, scale_each = True)

            pred_real = model(real)[0]
            pred_real = pred_real.squeeze(3).squeeze(2).cpu().numpy()
            real_embeddings.append(pred_real)
            pred_fake = model(fake)[0]
            pred_fake = pred_fake.squeeze(3).squeeze(2).cpu().numpy()
            fake_embeddings.append(pred_fake)

    real_embeddings = np.concatenate(real_embeddings, axis=0)
    real_mu = np.mean(real_embeddings, axis=0)
    real_sigma = np.cov(real_embeddings, rowvar=False)

    fake_embeddings = np.concatenate(fake_embeddings, axis=0)
    fake_mu = np.mean(fake_embeddings, axis=0)
    fake_sigma = np.cov(fake_embeddings, rowvar=False)

    print('Total number of samples:', real_embeddings.shape[0])
    
    fid_value = calculate_frechet_distance(real_mu, real_sigma, fake_mu, fake_sigma)
    return fid_value

def test_fid_ddpm_N(ema, vaemodel, mlp, coords, loader, accelerator, shape, total_fake_number, path=None, save=False):
    real_embeddings = []
    fake_embeddings = []

    device = accelerator.device    
    ema.ema_model.eval()
    vaemodel.eval()
    mlp.eval()
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx]).to(device)
    model.eval()
    size = coords.shape[2]
    si = get_scale_injection(size)
    with torch.inference_mode():
        for n, (real, idx) in enumerate(loader):
            batch_size = real.size(0)
            real = real.to(device)
            real = symmetrize_image_data(real)
            real = trans_F.resize(real, size, antialias = True)
            real = real.clamp(-1, 1)
            pred_real = model(real)[0]
            pred_real = pred_real.squeeze(3).squeeze(2).cpu().numpy()
            real_embeddings.append(pred_real)

    real_embeddings = np.concatenate(real_embeddings, axis=0)
    real_mu = np.mean(real_embeddings, axis=0)
    real_sigma = np.cov(real_embeddings, rowvar=False)

    batch_size = shape[0]
    iter = total_fake_number // batch_size
    with torch.inference_mode():
        for idx in range(iter):
            with accelerator.autocast():
                z_test = ema.ema_model.sample(shape = shape)
                if isinstance(vaemodel, torch.nn.parallel.DistributedDataParallel):
                    pe_test = vaemodel.module.decode(z_test)
                else:
                    pe_test = vaemodel.decode(z_test)
                fake = mlp(coords, hdbf=pe_test, si=si)
            fake = fake.clamp(-1., 1.)
            fake2 = (fake+1)/2
            
            if save:
                assert path is not None
                for k in range(fake.shape[0]):
                    fake_img = fake2[k].data.cpu().numpy().transpose(1,2,0)
                    fake_img = Image.fromarray((fake_img * 255).astype(np.uint8))
                    fake_img.save(os.path.join((path), 'gen-{}-{}.jpg'.format(idx, k)))

            pred_fake = model(fake)[0]
            pred_fake = pred_fake.squeeze(3).squeeze(2).cpu().numpy()
            fake_embeddings.append(pred_fake)

    fake_embeddings = np.concatenate(fake_embeddings, axis=0)
    fake_mu = np.mean(fake_embeddings, axis=0)
    fake_sigma = np.cov(fake_embeddings, rowvar=False)

    print('Total number of real samples:', real_embeddings.shape[0])
    print('Total number of real samples:', fake_embeddings.shape[0])
    
    fid_value = calculate_frechet_distance(real_mu, real_sigma, fake_mu, fake_sigma)
    return fid_value



### Video evaluation Metric

def test_rfvd(vaemodel, mlp, coords, loader, device, accelerator, logger=None):
    check = time.time()

    real_embeddings = []
    fake_embeddings = []
    fakes = []
    reals = []

    vaemodel.eval()
    mlp.eval()
    i3d = load_i3d_pretrained(device)

    with torch.inference_mode():
        for n, (real, idx) in enumerate(loader):
            if n > 512:
                break
            batch_size = real.size(0)
            clip_length = real.size(1)
            real = real.to(device)
            with accelerator.autocast():
                if isinstance(vaemodel, torch.nn.parallel.DistributedDataParallel):
                    xy, yt, xt = vaemodel.module.encode(rearrange(real / 127.5 - 1, 'b t c h w -> b c t h w'))
                    xy, yt, xt = vaemodel.module.decode(xy.sample(), yt.sample(), xt.sample())
                else:
                    xy, yt, xt = vaemodel.encode(rearrange(real / 127.5 - 1, 'b t c h w -> b c t h w'))
                    xy, yt, xt = vaemodel.decode(xy.sample(), yt.sample(), xt.sample())
                fake = mlp(coords, (xy, yt, xt))

            real = rearrange(real, 'b t c h w -> b t h w c') # videos
            fake = rearrange((fake.clamp(-1,1) + 1) * 127.5, 'b c t h w -> b t h w c', b=real.size(0))

            real = real.type(torch.uint8).cpu()
            fake = fake.type(torch.uint8)

            real_embeddings.append(get_fvd_logits(real.numpy(), i3d=i3d, device=device))
            fake_embeddings.append(get_fvd_logits(fake.cpu().numpy(), i3d=i3d, device=device))
            if len(fakes) < 16:
                reals.append(rearrange(real[0:1], 'b t h w c -> b c t h w'))
                fakes.append(rearrange(fake[0:1], 'b t h w c -> b c t h w'))

    vaemodel.train()
    mlp.train()

    reals = torch.cat(reals)
    fakes = torch.cat(fakes)

    real_embeddings = torch.cat(real_embeddings)
    fake_embeddings = torch.cat(fake_embeddings)
    print('Total number of samples:', real_embeddings.shape[0])
    
    fvd = frechet_distance(fake_embeddings.clone().detach(), real_embeddings.clone().detach())
    return fvd.item()

def test_fvd_ddpm(ema, vaemodel, mlp, coords, loader, accelerator, shape, path=None, save=False):
    device = accelerator.device
    real_embeddings = []
    fake_embeddings = []

    i3d = load_i3d_pretrained(device)
    
    with torch.inference_mode():
        for n, (real, idx) in enumerate(loader):
            #if n > 512:
            #    break
            real = rearrange(real, 'b t c h w -> b t h w c') # videos
            real = real.type(torch.uint8).cpu()
            real_embeddings.append(get_fvd_logits(real.numpy(), i3d=i3d, device=device))

            with accelerator.autocast():
                z_test = ema.ema_model.sample(shape=shape)
                if isinstance(vaemodel, torch.nn.parallel.DistributedDataParallel):
                    pe_test = vaemodel.module.decode(z_test)
                else:
                    pe_test = vaemodel.decode(z_test)
                fake = mlp(coords, pe_test)
            fake = rearrange((fake.clamp(-1,1) + 1) * 127.5, 'b c t h w -> b t h w c', b=shape[0])
            fake = fake.type(torch.uint8)
            fake_embeddings.append(get_fvd_logits(fake.cpu().numpy(), i3d=i3d, device=device))
    real_embeddings = torch.cat(real_embeddings)
    fake_embeddings = torch.cat(fake_embeddings)
    print('Total number of real samples:', real_embeddings.shape[0])
    print('Total number of real samples:', fake_embeddings.shape[0])

    fvd = frechet_distance(fake_embeddings.clone().detach(), real_embeddings.clone().detach())
    return fvd.item()

