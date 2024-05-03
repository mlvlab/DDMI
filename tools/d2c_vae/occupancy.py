"""
wild mixture of
https://github.com/autonomousvision/convolutional_occupancy_networks
for implementing occupancy function
"""

import os
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from torch import distributions as dist
from torchvision import utils as vtils
from torch.optim import AdamW
from accelerate import Accelerator

from utils.general_utils import exists, linear_kl_coeff
from utils.sr_utils import SpectralNormCalculator


class D2CTrainer(object):
    def __init__(
            self,
            args,
            pointnet,
            vaemodel,
            mlp,
            mesh_gen,
            data,
            ):
        super().__init__()

        ## Accelerator
        self.accelerator = Accelerator(
                split_batches = False,
                mixed_precision = 'fp16' if args.use_fp16 else 'no'
                )
        self.accelerator.native_amp = True if args.use_fp16 else False

        self.data = data
        self.args = args

        ## models
        self.pointnet = pointnet
        self.vaemodel = vaemodel
        self.mlp = mlp
        self.mesh_gen = mesh_gen

        ## loss config
        if args.loss_config.adversarial:
            raise NotImplementedError
        self.train_lr = args.lr
        self.warmup_epochs = args.loss_config.warmup_epochs
        self.epochs = args.loss_config.epochs
        self.save_and_sample_every = args.loss_config.save_and_sample_every
        self.warmup_iters = len(data) * self.warmup_epochs
        self.num_total_iters = len(data) * self.epochs
        self.kl_anneal_portion = args.loss_config.kl_anneal_portion
        self.kl_const_portion = args.loss_config.kl_const_portion
        self.kl_const_coeff = args.loss_config.kl_const_coeff
        self.kl_max_coeff = args.loss_config.kl_max_coeff
        self.sn_reg_weight_decay = args.loss_config.sn_reg_weight_decay
        self.sn_reg_weight_decay_anneal = args.loss_config.sn_reg_weight_decay_anneal
        self.sn_reg_weight_decay_init = args.loss_config.sn_reg_weight_decay_init
        self.sn_reg = args.loss_config.sn_reg
        self.gradient_accumulate_every = args.loss_config.gradient_accumulate_every
        if args.loss_config.sn_reg:
            print('Spectral Normalization Regularization Activated!')
            self.sn_caculator = SpectralNormCalculator()
            self.sn_caculator.add_conv_layers(vaemodel)
            self.sn_caculator.add_bn_layers(vaemodel)
        else:
            self.sn_caculator = None
            print('No SN Reg!')

        ## Optimizer
        self.opt = AdamW(list(pointnet.parameters()) + list(vaemodel.parameters()) + list(mlp.parameters()), lr = args.lr, betas = (0.9, 0.99))
        if args.loss_config.lr_scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.opt, (self.epochs - self.warmup_epochs - 1), eta_min = 1e-6
            )
        else:
            self.scheduler = None

        ## Initialize
        self.step = 0
        self.current_iters = 0
    
        ## Load from previous checkpoint
        if args.resume:
            print('Load checkpoint from previous training!')
            self.load(os.path.join(args.data_config.save_pth, 'model-last.pt'))
            print('Current Epochs : ', self.step)
            print('Current iters : ', self.current_iters)

        self.data, self.pointnet, self.vaemodel, self.mlp, self.opt = self.accelerator.prepare(self.data, self.pointnet, self.vaemodel, self.mlp, self.opt)

        if exists(self.sn_caculator):
            self.sn_caculator = self.accelerator.prepare(self.sn_caculator)

        ## Save directory
        self.results_folder = args.data_config.save_pth
        os.makedirs(self.results_folder, exist_ok=True)
        self.results_pth = os.path.join(self.results_folder, 'results')
        os.makedirs(self.results_pth, exist_ok=True)
        
    def save(self, step = 0):
        if not self.accelerator.is_main_process:
            return
        data = {
                'step' : self.step,
                'current_iters' : self.current_iters,
                'pointnet' : self.accelerator.get_state_dict(self.pointnet),
                'vaemodel' : self.accelerator.get_state_dict(self.vaemodel),
                'mlp' : self.accelerator.get_state_dict(self.mlp),
                'opt' : self.opt.state_dict(),
                'scaler' : self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
                'opt_sched' : self.scheduler.state_dict() if exists(self.scheduler) else None,
                'sn' : self.sn_caculator.state_dict() if exists(self.sn_caculator) else None
                }
        torch.save(data, os.path.join(self.results_folder, 'model-{}.pt'.format(step)))
        torch.save(data, os.path.join(self.results_folder, 'model-last.pt'.format(step)))
 
    def load(self, pth):
        data = torch.load(pth, map_location = 'cpu')
        self.pointnet.load_state_dict(data['pointnet'])
        self.vaemodel.load_state_dict(data['vaemodel'])
        self.mlp.load_state_dict(data['mlp'])
        self.step = data['step']
        self.current_iters = data['current_iters']
        self.opt.load_state_dict(data['opt'])

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

        if exists(self.scheduler) and exists(data['opt_sched']):
            self.scheduler.load_state_dict(data['opt_sched'])
        
        if exists(self.sn_caculator) and data['sn'] is not None:
            self.sn_caculator.load_state_dict(data['sn'], device = self.accelerator.device)

    def train(self):
        device = self.accelerator.device

        with tqdm(initial = self.step, total = self.epochs) as pbar:
            while self.step < self.epochs:
                if self.step < self.warmup_epochs:
                    lr = self.train_lr * float(self.step + 1) / self.warmup_epochs
                    for param_group in self.opt.param_groups:
                        param_group['lr'] = lr
                
                if self.step > self.warmup_epochs and self.scheduler is not None:
                    self.scheduler.step()

                self.pointnet.train()
                self.vaemodel.train()
                self.mlp.train()

                for idx, batch in enumerate(self.data):
                    coords = batch['points']
                    occ = batch['points.occ']
                    inputs = batch.get('inputs', torch.empty(coords.size(0), 0)).to(device)
                    
                    with self.accelerator.autocast():
                        ## pointclouds to grid planes using local point pooling network
                        f_planes = self.pointnet(inputs)
                        if isinstance(self.vaemodel, torch.nn.parallel.DistributedDataParallel):
                            posterior_xy, posterior_yz, posterior_xz = self.vaemodel.module.encode([f_planes['xy'], f_planes['yz'], f_planes['xz']])
                            xy, yz, xz = posterior_xy.sample(), posterior_yz.sample(), posterior_xz.sample()
                            z = torch.cat([xy, yz, xz], dim = 1)
                            pe_xy, pe_yz, pe_xz = self.vaemodel.module.decode(z)
                            
                        else:
                            posterior_xy, posterior_yz, posterior_xz = self.vaemodel.encode([f_planes['xy'], f_planes['yz'], f_planes['xz']])
                            xy, yz, xz = posterior_xy.sample(), posterior_yz.sample(), posterior_xz.sample()
                            z = torch.cat([xy, yz, xz], dim = 1)
                            pe_xy, pe_yz, pe_xz = self.vaemodel.decode((posterior_xy.sample(), posterior_yz.sample(), posterior_xz.sample()))

                        output = self.mlp(coords, (pe_xy, pe_yz, pe_xz))
                        logits = output.logits

                        ## Recon Loss
                        loss_i = F.binary_cross_entropy_with_logits(logits, occ, reduction = 'none')
                        recon_loss = loss_i.sum(-1).mean()

                        ## KL loss
                        kld_xy = posterior_xy.kl()
                        kld_yz = posterior_yz.kl()
                        kld_xz = posterior_xz.kl()
                        kld = kld_xy + kld_yz + kld_xz
                        kld_loss = torch.mean(kld)
                        kl_coeff = linear_kl_coeff(self.current_iters, self.kl_anneal_portion * self.num_total_iters,
                                                        self.kl_const_portion * self.num_total_iters, self.kl_const_coeff, self.kl_max_coeff)

                        total_loss = recon_loss + kl_coeff * kld_loss

                        # SN regularization
                        if self.sn_reg:
                            norm_loss = self.sn_caculator.spectral_norm_parallel()
                            bn_loss = self.sn_caculator.batchnorm_loss()
                            if self.sn_reg_weight_decay_anneal:
                                wdn_coeff = (1. - kl_coeff) * np.log(self.sn_reg_weight_decay_init) + kl_coeff * np.log(self.sn_reg_weight_decay)
                                wdn_coeff = np.exp(wdn_coeff)
                            else:
                                wdn_coeff = self.sn_reg_weight_decay
                            
                            total_loss += norm_loss * wdn_coeff + bn_loss * wdn_coeff
                    pbar.set_description('recon : {:.4f} \ kl : {:.4f}'.format(recon_loss, kld_loss.item()))
                    self.accelerator.backward(total_loss)
                    self.current_iters += 1
                    self.accelerator.wait_for_everyone()

                    if self.current_iters % self.gradient_accumulate_every == self.gradient_accumulate_every - 1:
                        self.opt.step()
                        self.opt.zero_grad()
                        
                        self.accelerator.wait_for_everyone()
                
                if self.step % self.save_and_sample_every == 0 and self.accelerator.is_main_process:
                    #print(logits.shape)
                    mesh, mesh2 = self.mesh_gen.generate_mesh_fromdiffusion((posterior_xy.sample()[0].unsqueeze(0), posterior_yz.sample()[0].unsqueeze(0), posterior_xz.sample()[0].unsqueeze(0)), self.vaemodel, self.mlp, self.accelerator.device)
                    mesh.export(os.path.join(self.results_pth, '{}.obj'.format(self.step)))
                    self.save(step = self.step)
                
                self.accelerator.wait_for_everyone()
                self.step += 1
                pbar.update(1)

    def eval(self):
        pass