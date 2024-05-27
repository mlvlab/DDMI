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
#from evals.eval import test_fid_ddpm, test_fid_ddpm_50k
from utils import nerf_helpers

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
            embed_fn,
            embeddirs_fn,
            cfg,
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
        self.embed_fn = embed_fn
        self.embeddirs_fn = embeddirs_fn
        self.cfg = cfg

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
        render_kwargs = nerf_helpers.get_render_kwargs(self.cfg, self.mlp, self.embed_fn, self.embeddirs_fn)
        noise_fix = torch.randn((self.test_batch_size, 3*self.channels, self.size1, self.size2), device = device)

        with tqdm(initial = self.step, total = self.epochs) as pbar:
            while self.step < self.epochs:
                for idx, (entry, cat, obj_path) in enumerate(self.data):
                    points = entry['data']
                    #import pdb;pdb.set_trace()
                    points = points.to(device, dtype=torch.float32)
                    gt_image = entry['images']
                    batch = gt_image.shape[0]
                    H = entry["images"][0].shape[1]
                    W = entry["images"][0].shape[2]
                    focal = .5 * W / np.tan(.5 * 0.6911112070083618) 
                    K = np.array([
                    [focal, 0, 0.5*W],
                    [0, focal, 0.5*H],
                    [0, 0, 1]
                    ])
                    pose_idx = np.random.choice(len(gt_image[0]), batch)
                    pose = entry['cam_poses'][0][pose_idx, :3, :4][0].to(device)
                    target = gt_image[:,pose_idx][0]
                    hw_idx = np.random.randint(0, H*W, 5000)
                    target_sort = target.reshape(target.shape[0], H*W, -1)[:, hw_idx]

                    with self.accelerator.autocast():
                        ## Encode latent
                        with torch.no_grad():
                            f_planes = self.pointnet(points)
                            x = torch.cat([f_planes['xy'], f_planes['yz'], f_planes['xz']], dim = 1)
                            if isinstance(self.vaemodel, torch.nn.parallel.DistributedDataParallel):
                                posterior_xy, posterior_yz, posterior_xz = self.vaemodel.module.encode([f_planes['xy'], f_planes['yz'], f_planes['xz']])
                                z = torch.cat([posterior_xy.sample(), posterior_yz.sample(), posterior_xz.sample()], dim = 1)
                            else:
                                posterior_xy, posterior_yz, posterior_xz = self.vaemodel.encode([f_planes['xy'], f_planes['yz'], f_planes['xz']])
                                z = torch.cat([posterior_xy.sample(), posterior_yz.sample(), posterior_xz.sample()], dim = 1)

                        ## LDM
                        z = z.detach()
                        p_loss,_ = self.diffusion_process(z)
                        p_loss = p_loss / self.gradient_accumulate_every

                    self.accelerator.backward(p_loss)
                    self.current_iters += 1

                    pbar.set_description('Dae loss : {:.3f}'.format(p_loss.item()))

                    self.accelerator.wait_for_everyone()

                    if self.current_iters % self.gradient_accumulate_every == self.gradient_accumulate_every - 1:
                        #if self.accelerator.sync_gradients:
                        #    self.accelerator.clip_grad_norm_(self.diffusion_process.parameters(), 1.)
                        self.dae_opt.step()
                        self.dae_opt.zero_grad()
                        self.accelerator.wait_for_everyone()
                        if self.accelerator.is_main_process:
                            self.ema.update()

                if self.step % self.save_and_sample_every == 0 and self.accelerator.is_main_process: 
                    _shape = z.shape[1:] 
                    shape = (self.test_batch_size, *_shape)
                    self.ema.ema_model.eval()
                    
                    #with self.accelerator.autocast():
                    with torch.inference_mode():
                        z_test = self.ema.ema_model.sample(shape=shape, noise = noise_fix)
                        pe = self.vaemodel.module.decode(z_test)
                        fea = {}
                        fea['xy'] = pe[0][0]
                        fea['yz'] = pe[1][0]
                        fea['xz'] = pe[2][0]
                        output = nerf_helpers.render(H, W, K, fea, None, 0, device, chunk=4096, c2w=pose,
                                                                verbose=True, retraw=True, hw_idx=None,
                                                                **render_kwargs)
                    output = output.unsqueeze(0).reshape(1, 200, 200, 3)
                    output_img = output.permute(0, 3, 1, 2)
                    vtils.save_image(output_img, os.path.join(self.results_pth, '{}.png'.format(self.current_iters)), normalize=True, scale_each=True)
                    

                if self.step % 100 == 0 and self.accelerator.is_main_process: 
                    self.save(step = self.step)

                self.accelerator.wait_for_everyone()
                self.step += 1
                pbar.update(1)

    def eval(self):
        raise NotImplementedError

    @torch.no_grad()
    def generate(self):
        shape = [1, 3*self.channels, self.size1, self.size2]
        render_kwargs = nerf_helpers.get_render_kwargs(self.cfg, self.mlp, self.embed_fn, self.embeddirs_fn)
        render_kwargs['perturb'] = False
        render_kwargs['raw_noise_std'] = 0.

        render_iterations = 10
        H, W = self.test_resolution, self.test_resolution
        focal = .5 * W / np.tan(.5 * 0.6911112070083618) 
        K = np.array([
                [focal, 0, 0.5*W],
                [0, focal, 0.5*H],
                [0, 0, 1]
                ])
        
        render_pose = torch.stack([nerf_helpers.pose_spherical(angle, -20, 5).to(self.accelerator.device) for angle in np.linspace(-180,180, render_iterations)[:-1]])
                                  
        for i in range(self.test_batch_size):
            z_test = self.ema.ema_model.sample(shape=shape)
            pe = self.vaemodel.module.decode(z_test)
            fea = {}
            fea['xy'] = pe[0][0]
            fea['yz'] = pe[1][0]
            fea['xz'] = pe[2][0]

            for k, pose in enumerate(render_pose):
                output = nerf_helpers.render(H, W, K, fea, None, 0, self.accelerator.device, chunk=4096, c2w=pose,
                                                        verbose=True, retraw=True, hw_idx=None,
                                                        **render_kwargs)
                output = output.unsqueeze(0).reshape(1, H, W, 3)
                output_img = output.permute(0, 3, 1, 2)
                vtils.save_image(output_img, os.path.join(self.results_pth, '{}-{}.png'.format(i ,k)), normalize=True, scale_each=True)
