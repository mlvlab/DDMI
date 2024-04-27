import os
import torch
import torchvision
import numpy as np
import copy
from tqdm import tqdm
from torchvision import utils as vtils
import torchvision.transforms.functional as trans_F
from timeit import default_timer as timer
from accelerate import Accelerator
from ema_pytorch import EMA

from utils.general_utils import symmetrize_image_data, unsymmetrize_image_data, exists, convert_to_coord_format_3d
from evals.eval import test_fvd_ddpm, test_fvd_ddpm_50k


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
        self.size1 = args.unetconfig.size1
        self.size2 = args.unetconfig.size2
        self.size3 = args.unetconfig.size3
        self.test_batch_size = args.data_config.test_batch_size
        self.channels = args.embed_dim
        self.num_total_iters = len(data) * self.epochs
        self.gradient_accumulate_every = args.loss_config.gradient_accumulate_every
        self.test_resolution = args.data_config.test_resolution

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
        shape = [self.test_batch_size, self.channels, self.size1*self.size2 + self.size1*self.size3 + self.size2*self.size3]
        noise_fix = torch.randn((self.test_batch_size, self.channels, self.size1*self.size2 + self.size1*self.size3 + self.size2*self.size3), device = device)

        with tqdm(initial = self.step, total = self.epochs) as pbar:
            while self.step < self.epochs:
                for idx, (x, _) in enumerate(self.data):
                    assert len(x.shape) == 5
                    x = (x / 127.5) - 1
                    x = x.permute(0, 2, 1, 3, 4).contiguous()
                    b, c, t, h, w = x.shape

                    with self.accelerator.autocast():
                        ## Encode latent
                        with torch.no_grad():
                            if isinstance(self.vaemodel, torch.nn.parallel.DistributedDataParallel):
                                posterior_xy, posterior_yt, posterior_xt = self.vaemodel.module.encode(x)
                            else:
                                posterior_xy, posterior_yt, posterior_xt = self.vaemodel.encode(x)

                        ## LDM
                        b, c = posterior_xy.size(0), posterior_xy.size(1)
                        z = torch.cat([posterior_xy.reshape(b, c, -1), posterior_xt.reshape(b, c, -1), posterior_yt.reshape(b, c, -1)], dim = 2)
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
                    coords = convert_to_coord_format_3d(1, 256, 256, 16, device = device, hstart=-255/256, hend=255/256, 
                                            wstart=-255/256, wend=255/256, tstart = -15/16, tend = 15/16)
                    self.ema.ema_model.eval()
                    with self.accelerator.autocast():
                        with torch.inference_mode():
                            z_test = self.ema.ema_model.sample(shape=shape, noise = noise_fix)
                            z_xy = z_test[:, :, 0:self.size1*self.size2].view(z_test.size(0), z_test.size(1), self.size1, self.size2)
                            z_xt = z_test[:, :, self.size1*self.size2:self.size1*(self.size2+self.size3)].view(z.test.size(0), z_test.size(1), self.size3, self.size2)
                            z_yt = z_test[:, :, self.size1*(self.size2+self.size3):self.size1*(self.size2+self.size3+self.size3)].view(z_test.size(0), z_test.size(1), self.size3, self.size2)
                            if isinstance(self.vaemodel, torch.nn.parallel.DistributedDataParallel):
                                pe_test = self.vaemodel.module.decode((z_xy, z_yt, z_xt))
                            else:
                                pe_test = self.vaemodel.decode((z_xy, z_yt, z_xt))
                            output_img = self.mlp(coords, pe_test)
                    output_img = output_img.clamp(min = -1., max = 1.)

                    step_save_pth = os.path.join(self.results_pth, 'step{}'.format(self.step))
                    os.makedirs(step_save_pth, exist_ok=True)
                    for ci in range(output_img.shape[2]):
                        vtils.save_image(output_img[:,:,ci], os.path.join(step_save_pth, 'gen-{}-{}.png'.format(ci, self.step)), normalize = False, scale_each = False)
                    
                    self.save(step = self.step)
                
                if self.step % 100 == 0 and self.accelerator.is_main_process and self.step > 300:
                    if self.test_data is not None:
                        coords = convert_to_coord_format_3d(1, 256, 256, 16, device = device, hstart=-255/256, hend=255/256, 
                                            wstart=-255/256, wend=255/256, tstart = -15/16, tend = 15/16)
                        fid = test_fvd_ddpm(self.ema, self.vaemodel, self.mlp, coords, self.test_data, self.accelerator, shape=[self.channels, self.size1, self.size2, self.size3])
                        print('Step {} FID: {}'.format(self.step, fid))
                    else:
                        self.accelerator.print('Not found test dataset to evaluate!')

                self.accelerator.wait_for_everyone()
                self.step += 1
                pbar.update(1)


    def eval(self):
        print('Evaluation!')
        device = self.accelerator.device
        coords = convert_to_coord_format_3d(1, 256, 256, 16, device = device, hstart=-255/256, hend=255/256, 
                                            wstart=-255/256, wend=255/256, tstart = -15/16, tend = 15/16)
    
        if 0:
            self.ema.ema_model.eval()
            with self.accelerator.autocast():
                with torch.inference_mode():
                    z_test = self.ema.ema_model.sample(batch_size = self.test_batch_size) / self.vae_scale_factor
                    if isinstance(self.vaemodel, torch.nn.parallel.DistributedDataParallel):
                        pe_test = self.vaemodel.module.decode(z_test)
                    else:
                        pe_test = self.vaemodel.decode(z_test)
                    output_img = self.mlp(coords, hdbf=pe_test, si=1)
            output_img = output_img.clamp(min = -1., max = 1.)
            vtils.save_image(output_img, os.path.join(self.results_pth, 'Test-{}.png'.format(self.step)), normalize = True, scale_each = True)
        
        if 0:
            fid = test_fid_ddpm_50k(self.ema, self.vaemodel, self.mlp, coords, self.data, self.accelerator, self.results_pth)
            print('Step {} FID: {}'.format(self.step, fid))
        
        if self.test_data is not None:
            fid = test_fid_ddpm(self.ema, self.vaemodel, self.mlp, coords, self.test_data, self.accelerator, self.results_pth, save=True)
            print('Step {} FID: {}'.format(self.step, fid))
        else:
            self.accelerator.print('Not found test dataset to evaluate!')