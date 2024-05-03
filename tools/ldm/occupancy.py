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
from models.ema import LitEma

from utils.general_utils import exists

# Trainer class
class LDMTrainer(object):
    def __init__(
            self,
            args,
            pointnet,
            vaemodel,
            mlp,
            diffusionmodel,
            diffusion_process,
            data,
            mesh_gen,
            test_data=None,
            ):
        super().__init__()

        ## Accelerator
        self.accelerator = Accelerator(
                split_batches = False,
                mixed_precision = 'fp16' if args.use_fp16 else 'no'
                )
        self.accelerator.native_amp = True if args.use_fp16 else False
        
        self.data = data
        self.test_data = test_data
        self.args = args

        # Models
        self.pointnet = pointnet
        self.vaemodel = vaemodel
        self.mlp = mlp
        self.diffusionmodel = diffusionmodel
        self.mesh_gen = mesh_gen

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
        elif args.pretrained:
            print('Loading Pretrained Models!')
            data_pth = torch.load(os.path.join(args.data_config.save_pth, 'ldm-last.pt'), map_location='cpu')
            self.pointnet.load_state_dict(data_pth['pointnet'])
            self.vaemodel.load_state_dict(data_pth['vaemodel'])
            self.mlp.load_state_dict(data_pth['mlp'])
            self.diffusion_process.load_state_dict(data_pth['diffusion'])
            if self.accelerator.is_main_process:
                self.ema.load_state_dict(data_pth['ema'])
        else:
            # Load from checkpoint
            print('Load VAE checkpoints!')
            data_pth = torch.load(os.path.join(args.data_config.save_pth, 'model-last.pt'), map_location='cpu')
            self.pointnet.load_state_dict(data_pth['pointnet'])
            self.vaemodel.load_state_dict(data_pth['vaemodel'])
            self.mlp.load_state_dict(data_pth['mlp'])

        # Wrap with accelerator
        self.data, self.pointnet, self.vaemodel, self.mlp, self.diffusion_process, self.dae_opt = self.accelerator.prepare(self.data, self.pointnet, self.vaemodel, self.mlp, self.diffusion_process, self.dae_opt)

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
                'pointnet' : self.accelerator.get_state_dict(self.pointnet),
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
        self.pointnet.load_state_dict(data['pointnet'])
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
        self.pointnet.eval()
        self.diffusion_process.train()
        noise_fix = torch.randn((self.test_batch_size, 3*self.channels, self.size1, self.size2), device = device)

        with tqdm(initial = self.step, total = self.epochs) as pbar:
            while self.step < self.epochs:
                for idx, batch in enumerate(self.data):
                    coords = batch['points']
                    occ = batch['points.occ']
                    inputs = batch.get('inputs', torch.empty(coords.size(0), 0)).to(device)

                    with self.accelerator.autocast():
                        ## Encode latent
                        with torch.no_grad():
                            f_planes = self.pointnet(inputs)
                            if isinstance(self.vaemodel, torch.nn.parallel.DistributedDataParallel):
                                posterior_xy, posterior_yz, posterior_xz = self.vaemodel.module.encode([f_planes['xy'], f_planes['yz'], f_planes['xz']])
                            else:
                                posterior_xy, posterior_yz, posterior_xz = self.vaemodel.encode([f_planes['xy'], f_planes['yz'], f_planes['xz']])

                        ## LDM
                        z_xy, z_yz, z_xz = posterior_xy.sample(), posterior_yz.sample(), posterior_xz.sample()
                        z = torch.cat([z_xy, z_xz, z_yz], dim = 1)
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
                    _shape = z.shape[1:] 
                    shape = (self.test_batch_size, *_shape)
                    self.ema.ema_model.eval()
                    
                    with self.accelerator.autocast():
                        with torch.inference_mode():
                            z_test = self.ema.ema_model.sample(shape=shape, noise = noise_fix)
                            mesh, mesh2 = self.mesh_gen.generate_mesh_fromdiffusion(z_test, self.vaemodel, self.mlp, self.accelerator.device)
                    mesh.export(os.path.join(self.results_pth, '{}.obj'.format(self.step)))
                    

                if self.step % 100 == 0 and self.accelerator.is_main_process: 
                    self.save(step = self.step)

                self.accelerator.wait_for_everyone()
                self.step += 1
                pbar.update(1)


    @torch.no_grad()
    def eval(self):
        print('Generating 5K shapes for evaluation!')
        shape = [self.test_batch_size, 3*self.channels, self.size1, self.size2]
        total_generation_number = 5000
        total_iters = total_generation_number // self.test_batch_size
        self.results_eval = os.path.join(self.results_folder, 'eval')
        os.makedirs(self.results_eval, exist_ok=True)

        for i in range(total_iters):
            print(i)
            z_test = self.ema.ema_model.sample(shape=shape)
            for j in range(z_test.shape[0]):
                mesh, mesh2 = self.mesh_gen.generate_mesh_fromdiffusion(z_test[j].unsqueeze(0), self.vaemodel, self.mlp, self.accelerator.device)    
                mesh.export(os.path.join(self.results_eval, '{}-{}.obj'.format(i,j)))
        print('Finished generating shapes!')
    
    @torch.no_grad()
    def generate(self):
        print('Generating shape!')
        shape = [1, 3*self.channels, self.size1, self.size2]
        with self.accelerator.autocast():
            z_test = self.ema.ema_model.sample(shape=shape)
            mesh, mesh2 = self.mesh_gen.generate_mesh_fromdiffusion(z_test, self.vaemodel, self.mlp, self.accelerator.device)
        mesh.export(os.path.join(self.results_pth, 'generation.obj'))
        print('Finished generating shapes!')
