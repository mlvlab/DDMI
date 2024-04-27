import os
import torch
import torchvision
import numpy as np
import copy
from PIL import Image
from tqdm import tqdm
from torchvision import utils as vtils
import torchvision.transforms.functional as trans_F
from timeit import default_timer as timer
from accelerate import Accelerator
from ema_pytorch import EMA


from models.ddmi import DDMI
from utils.general_utils import symmetrize_image_data, unsymmetrize_image_data, exists, convert_to_coord_format_2d
from evals.eval import test_fid_ddpm, test_fid_ddpm_50k

# Trainer class
class LDMTrainer(object):
    def __init__(
            self,
            args,
            vaemodel,
            mlp,
            diffusionmodel,
            diffusion_process,
            data,
            test_data=None,
            ):
        super().__init__()

        ## Accelerator
        self.accelerator = Accelerator(
                split_batches = False,
                mixed_precision = 'fp16' if args.use_fp16 else 'no'
                )
        self.accelerator.native_amp = args.amp
        
        self.data = data
        self.test_data = test_data
        self.args = args

        # Models
        self.vaemodel = vaemodel
        self.mlp = mlp
        self.diffusionmodel = diffusionmodel

        # Diffusion process
        self.diffusion_process = diffusion_process
        
        self.epochs = args.loss_config.epochs
        self.save_and_sample_every = args.loss_config.save_and_sample_every
        self.latent_dim = args.ddpmconfig.channels
        self.image_size = args.ddpmconfig.image_size
        self.test_batch_size = args.data_config.test_batch_size
        self.channels = args.embed_dim
        self.num_total_iters = len(data) * self.epochs
        self.gradient_accumulate_every = args.loss_config.gradient_accumulate_every
        self.test_resolution = args.data_config.test_resolution
        self.vae_scale_factor = 1.0

        # Optimizers
        self.dae_opt = torch.optim.AdamW(diffusion_process.parameters(), lr = args.lr)

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_process, beta = args.loss_config.ema_decay, update_every = args.loss_config.ema_update_every)
            self.ema.to(self.accelerator.device)
        # Reset epochs and iters
        self.step = 0
        self.current_iters = 0

        if args.resume:
            print('Loading Models from previous training!')
            self.load(os.path.join(args.data_config.save_pth, 'ldm-last.pt'))
            print('Current Epochs :', self.step)
            print('Current iters :', self.current_iters)
        else:
            # Load from checkpoint
            print('Load VAE checkpoints!')
            data_pth = torch.load(os.path.join(args.data_config.save_pth, 'model-last.pt'), map_location='cpu')
            self.vaemodel.load_state_dict(data_pth['model'])
            self.mlp.load_state_dict(data_pth['mlp'])

        # Wrap with accelerator
        self.data, self.vaemodel, self.mlp, self.diffusion_process, self.dae_opt = self.accelerator.prepare(self.data, self.vaemodel, self.mlp, self.diffusion_process, self.dae_opt)

        ## Save directory
        self.results_folder = args.data_config.save_pth
        os.makedirs(self.results_folder, exist_ok=True)
        self.results_pth = os.path.join(self.results_folder, 'results')
        os.makedirs(self.results_pth, exist_ok=True)
       
    def save(self, step = 0):
        if not self.accelerator.is_local_main_process:
            return
        data = {
                'args' : self.args,
                'step' : self.step,
                'current_iters' : self.current_iters,
                'vaemodel' : self.accelerator.get_state_dict(self.vaemodel),
                'mlp' : self.accelerator.get_state_dict(self.mlp),
                'diffusion' : self.accelerator.get_state_dict(self.diffusion_process),
                'dae_opt' : self.dae_opt.state_dict(),
                'ema' : self.ema.state_dict(),
                'scaler' : self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
                }
        torch.save(data, os.path.join(self.results_folder, 'ldm-{}.pt'.format(step)))
        torch.save(data, os.path.join(self.results_folder, 'ldm-last.pt'.format(step)))


    def load(self, pth):
        data = torch.load(pth, map_location= 'cpu')
        self.diffusion_process.load_state_dict(data['diffusion'])
        self.vaemodel.load_state_dict(data['vaemodel'])
        self.mlp.load_state_dict(data['mlp'])
        self.step = data['step']
        self.current_iters = data['current_iters']
        self.dae_opt.load_state_dict(data['dae_opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data['ema'])

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        device = self.accelerator.device
        self.vaemodel.eval()
        self.mlp.eval()
        self.diffusion_process.train()
        noise_fix = torch.randn((self.test_batch_size, self.channels, self.image_size, self.image_size), device = device)

        with tqdm(initial = self.step, total = self.epochs) as pbar:
            while self.step < self.epochs:
                for idx, (x, _) in enumerate(self.data):
                    x = symmetrize_image_data(x)
                    y = x

                    with self.accelerator.autocast():
                        ## Encode latent
                        with torch.no_grad():
                            if isinstance(self.vaemodel, torch.nn.parallel.DistributedDataParallel):
                                z = self.vaemodel.module.encode(y).sample() * self.vae_scale_factor
                            else:
                                z = self.vaemodel.encode(y).sample()

                        ## LDM
                        z = z.detach()
                        p_loss,_ = self.diffusion_process(z)
                        p_loss = p_loss / self.gradient_accumulate_every

                    self.accelerator.backward(p_loss)
                    self.current_iters += 1

                    pbar.set_description('Dae loss : {:.3f}'.format(p_loss.item()))

                    self.accelerator.wait_for_everyone()

                    if self.current_iters % self.gradient_accumulate_every == self.gradient_accumulate_every - 1:
                        self.dae_opt.step()
                        self.dae_opt.zero_grad()
                        self.accelerator.wait_for_everyone()
                        if self.accelerator.is_main_process:
                            self.ema.update()

                if self.step % self.save_and_sample_every == 0 and self.accelerator.is_main_process: 
                    coords = convert_to_coord_format_2d(1, 256, 256, device = device, hstart=-255/256, hend = 255/256, wstart=-255/256, wend = 255/256)
                    self.ema.ema_model.eval()
                    with self.accelerator.autocast():
                        with torch.inference_mode():
                            z_test = self.ema.ema_model.sample(batch_size = self.test_batch_size, noise = noise_fix) / self.vae_scale_factor
                            if isinstance(self.vaemodel, torch.nn.parallel.DistributedDataParallel):
                                pe_test = self.vaemodel.module.decode(z_test)
                            else:
                                pe_test = self.vaemodel.decode(z_test)
                            output_img = self.mlp(coords, hdbf=pe_test, si=1)
                    output_img = output_img.clamp(min = -1., max = 1.)
                    output_img = unsymmetrize_image_data(output_img)

                    vtils.save_image(output_img, os.path.join(self.results_pth, '{}.png'.format(self.step)), normalize = False, scale_each = False)
                    self.save(step = self.step)
                
                if self.step % 100 == 0 and self.accelerator.is_main_process and self.step > 300:
                    if self.test_data is not None:
                        coords = convert_to_coord_format_2d(1, 256, 256, device = device, hstart=-255/256, hend = 255/256, wstart=-255/256, wend = 255/256)
                        fid = test_fid_ddpm(self.ema, self.vaemodel, self.mlp, coords, self.test_data, self.accelerator)
                        print('Step {} FID: {}'.format(self.step, fid))
                    else:
                        self.accelerator.print('Not found test dataset to evaluate!')

                self.accelerator.wait_for_everyone()
                self.step += 1
                pbar.update(1)


    def eval(self):
        print('Start Generation')
        os.makedirs(self.args.data_config.save_gen, exist_ok=True)
        device = self.accelerator.device
        coords = convert_to_coord_format_2d(1, self.test_resolution, 
                                            self.test_resolution, 
                                            device = device, 
                                            hstart=-(self.test_resolution-1)/self.test_resolution, 
                                            hend = (self.test_resolution-1)/self.test_resolution, 
                                            wstart=-(self.test_resolution-1)/self.test_resolution, 
                                            wend = (self.test_resolution-1)/self.test_resolution)
        
        
        shape = [self.test_batch_size, self.channels, self.image_size, self.image_size]
        iters = self.args.data_config.num_samples // self.test_batch_size
        self.ema.ema_model.eval()

        for idx in range(iters):
            with self.accelerator.autocast():
                with torch.inference_mode():
                    z_test = self.ema.ema_model.sample(batch_size = self.test_batch_size) / self.vae_scale_factor
                    if isinstance(self.vaemodel, torch.nn.parallel.DistributedDataParallel):
                        pe_test = self.vaemodel.module.decode(z_test)
                    else:
                        pe_test = self.vaemodel.decode(z_test)
                    output_img = self.mlp(coords, hdbf=pe_test, si=1)
            output_img = output_img.clamp(min = -1., max = 1.)
            fake = unsymmetrize_image_data(output_img)
            for k in range(fake.shape[0]):
                    fake_img = fake[k].data.cpu().numpy().transpose(1,2,0)
                    fake_img = Image.fromarray((fake_img * 255).astype(np.uint8))
                    fake_img.save(os.path.join(self.args.data.config.save_gen, 'gen-{}-{}.jpg'.format(idx, k)))
